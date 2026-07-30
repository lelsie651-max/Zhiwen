from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import tempfile
from dataclasses import dataclass
from typing import BinaryIO

from charset_normalizer import from_bytes
from docx import Document as DocxDocumentFactory
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.styles.style import BaseStyle
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.schemas.document_extraction import (
    ExtractedBlock,
    ExtractedBlockType,
    ExtractedDocument,
    ExtractionOutcome,
)
from app.schemas.file_ingestion import DetectedFileFormat
from app.storage import FileStorage

logger = logging.getLogger(__name__)

CHUNK_READ_BYTES = 64 * 1024
SPOOL_MAX_SIZE = 4 * 1024 * 1024
MAX_BLOCK_CHARACTERS = 20_000
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
MARKDOWN_UNORDERED_LIST_PATTERN = re.compile(r"^\s*[-+*]\s+(.+)$")
MARKDOWN_ORDERED_LIST_PATTERN = re.compile(r"^\s*\d+\.\s+(.+)$")
MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
DOCX_HEADING_STYLE_ID_PATTERN = re.compile(r"^Heading([1-6])$", flags=re.IGNORECASE)
DOCX_HEADING_STYLE_NAME_PATTERN = re.compile(r"^Heading\s*([1-6])$", flags=re.IGNORECASE)
MARKDOWN_FENCE_PATTERN = re.compile(r"^(?P<indent>[ ]{0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$")


@dataclass(frozen=True)
class _RawBlock:
    block_type: ExtractedBlockType
    raw_text: str
    location_base: str
    block_index: int
    page_no: int | None = None
    heading_level: int | None = None
    heading_path: list[str] | None = None
    start_line: int | None = None
    end_line: int | None = None
    table_index: int | None = None
    row_index: int | None = None
    metadata: dict[str, object] | None = None
    collapse_spaces: bool = True


async def extract_document(
    storage: FileStorage,
    storage_key: str,
    detected_format: DetectedFileFormat | str,
    encoding_hint: str | None = None,
) -> ExtractedDocument:
    file_format = (
        detected_format
        if isinstance(detected_format, DetectedFileFormat)
        else DetectedFileFormat(detected_format)
    )
    spool_file = tempfile.SpooledTemporaryFile(max_size=SPOOL_MAX_SIZE, mode="w+b")

    try:
        await _stream_storage_to_spool(storage, storage_key, spool_file)
        if file_format == DetectedFileFormat.PDF:
            return await asyncio.to_thread(_extract_pdf, spool_file)
        if file_format == DetectedFileFormat.DOCX:
            return await asyncio.to_thread(_extract_docx, spool_file)
        if file_format == DetectedFileFormat.MD:
            return await asyncio.to_thread(_extract_markdown, spool_file, encoding_hint)
        if file_format == DetectedFileFormat.TXT:
            return await asyncio.to_thread(_extract_txt, spool_file, encoding_hint)
    except Exception:
        logger.warning("Document extraction failed without exposing storage details.")
        return ExtractedDocument(
            outcome=ExtractionOutcome.FAILED,
            detected_format=file_format,
            detected_encoding=None,
            page_count=None,
            character_count=0,
            block_count=0,
            blocks=[],
            warnings=["提取失败，当前文件无法安全读取。"],
            metadata={},
        )
    finally:
        spool_file.close()

    raise ValueError("Unsupported detected format.")


async def _stream_storage_to_spool(
    storage: FileStorage,
    storage_key: str,
    spool_file: BinaryIO,
) -> None:
    async for chunk in storage.read_file_chunks(storage_key, chunk_size=CHUNK_READ_BYTES):
        spool_file.write(chunk)
    spool_file.seek(0)


def _extract_pdf(spool_file: BinaryIO) -> ExtractedDocument:
    try:
        reader = PdfReader(spool_file)
    except PdfReadError:
        return _build_document(
            outcome=ExtractionOutcome.FAILED,
            detected_format=DetectedFileFormat.PDF,
            detected_encoding=None,
            page_count=None,
            raw_blocks=[],
            warnings=["PDF 文件损坏或无法解析。"],
            metadata={},
        )

    if reader.is_encrypted:
        return _build_document(
            outcome=ExtractionOutcome.FAILED,
            detected_format=DetectedFileFormat.PDF,
            detected_encoding=None,
            page_count=len(reader.pages),
            raw_blocks=[],
            warnings=["加密 PDF 当前无法提取正文。"],
            metadata={"empty_page_count": len(reader.pages)},
        )

    raw_blocks: list[_RawBlock] = []
    warnings: list[str] = []
    empty_page_count = 0
    block_index = 0

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = _normalize_newlines(page_text)
        if not page_text.strip():
            warnings.append(f"PDF 第 {page_number} 页没有可提取文本。")
            empty_page_count += 1
            continue

        for page_block_index, text_block in enumerate(_split_text_blocks(page_text)):
            raw_blocks.append(
                _RawBlock(
                    block_type=ExtractedBlockType.PAGE_TEXT,
                    raw_text=text_block,
                    location_base=f"pdf:p{page_number}:b{page_block_index}",
                    block_index=block_index,
                    page_no=page_number,
                    metadata={},
                )
            )
            block_index += 1

    if not raw_blocks:
        return _build_document(
            outcome=ExtractionOutcome.NEEDS_OCR,
            detected_format=DetectedFileFormat.PDF,
            detected_encoding=None,
            page_count=len(reader.pages),
            raw_blocks=[],
            warnings=warnings or ["PDF 没有可提取文本，后续需要 OCR。"],
            metadata={"empty_page_count": empty_page_count},
        )

    outcome = ExtractionOutcome.PARTIAL if empty_page_count > 0 else ExtractionOutcome.SUCCESS
    return _build_document(
        outcome=outcome,
        detected_format=DetectedFileFormat.PDF,
        detected_encoding=None,
        page_count=len(reader.pages),
        raw_blocks=raw_blocks,
        warnings=warnings,
        metadata={"empty_page_count": empty_page_count},
    )


def _extract_docx(spool_file: BinaryIO) -> ExtractedDocument:
    document = DocxDocumentFactory(spool_file)
    raw_blocks: list[_RawBlock] = []
    warnings: list[str] = []
    heading_path: list[str] = []
    block_index = 0
    paragraph_index = 0
    table_index = 0

    for element in document.element.body.iterchildren():
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, document)
            text = _normalize_newlines(paragraph.text)
            if not text.strip():
                paragraph_index += 1
                continue

            style = paragraph.style
            style_name = _get_docx_style_name(style)
            block_type = ExtractedBlockType.PARAGRAPH
            heading_level: int | None = None
            block_heading_path = list(heading_path)

            heading_level = _get_docx_heading_level(style)
            if heading_level is not None:
                block_type = ExtractedBlockType.HEADING
                heading_path = heading_path[: heading_level - 1] + [text.strip()]
                block_heading_path = list(heading_path)
            elif _is_list_paragraph(paragraph, style_name):
                block_type = ExtractedBlockType.LIST_ITEM

            raw_blocks.append(
                _RawBlock(
                    block_type=block_type,
                    raw_text=text,
                    location_base=f"docx:p{paragraph_index}",
                    block_index=block_index,
                    heading_level=heading_level,
                    heading_path=block_heading_path,
                    metadata={},
                )
            )
            paragraph_index += 1
            block_index += 1
            continue

        if isinstance(element, CT_Tbl):
            table = Table(element, document)
            for row_position, row in enumerate(table.rows):
                cell_texts = [_normalize_newlines(cell.text) for cell in row.cells]
                if not any(cell.strip() for cell in cell_texts):
                    continue

                raw_blocks.append(
                    _RawBlock(
                        block_type=ExtractedBlockType.TABLE_ROW,
                        raw_text=" | ".join(cell_texts),
                        location_base=f"docx:t{table_index}:r{row_position}",
                        block_index=block_index,
                        heading_path=list(heading_path),
                        table_index=table_index + 1,
                        row_index=row_position + 1,
                        metadata={"cells": cell_texts},
                    )
                )
                block_index += 1
            table_index += 1

    if not raw_blocks:
        warnings.append("DOCX 没有可提取的正文或表格文本。")
        return _build_document(
            outcome=ExtractionOutcome.FAILED,
            detected_format=DetectedFileFormat.DOCX,
            detected_encoding=None,
            page_count=None,
            raw_blocks=[],
            warnings=warnings,
            metadata={"image_count": len(document.inline_shapes)},
        )

    return _build_document(
        outcome=ExtractionOutcome.SUCCESS,
        detected_format=DetectedFileFormat.DOCX,
        detected_encoding=None,
        page_count=None,
        raw_blocks=raw_blocks,
        warnings=warnings,
        metadata={"image_count": len(document.inline_shapes)},
    )


def _extract_markdown(spool_file: BinaryIO, encoding_hint: str | None) -> ExtractedDocument:
    decoded_text, detected_encoding = _decode_text_bytes(spool_file, encoding_hint)
    if decoded_text is None:
        return _build_document(
            outcome=ExtractionOutcome.FAILED,
            detected_format=DetectedFileFormat.MD,
            detected_encoding=None,
            page_count=None,
            raw_blocks=[],
            warnings=["Markdown 文件编码无法识别。"],
            metadata={},
        )

    lines = _normalize_newlines(decoded_text).split("\n")
    raw_blocks: list[_RawBlock] = []
    heading_path: list[str] = []
    warnings: list[str] = []
    block_index = 0
    line_index = 0
    table_index = 0
    code_index = 0

    while line_index < len(lines):
        line_number = line_index + 1
        line = lines[line_index]

        if not line.strip():
            line_index += 1
            continue

        heading_match = MARKDOWN_HEADING_PATTERN.match(line)
        if heading_match:
            heading_level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_path = heading_path[: heading_level - 1] + [heading_text]
            raw_blocks.append(
                _RawBlock(
                    block_type=ExtractedBlockType.HEADING,
                    raw_text=line,
                    location_base=f"md:h{line_number}",
                    block_index=block_index,
                    heading_level=heading_level,
                    heading_path=list(heading_path),
                    start_line=line_number,
                    end_line=line_number,
                    metadata={},
                )
            )
            block_index += 1
            line_index += 1
            continue

        fence_match = _parse_markdown_fence_start(line)
        if fence_match is not None:
            code_block, next_line_index, warning = _consume_markdown_fenced_code_block(
                lines,
                line_index,
                block_index,
                code_index,
                heading_path,
                fence_match,
            )
            raw_blocks.append(code_block)
            if warning is not None:
                warnings.append(warning)
            code_index += 1
            block_index += 1
            line_index = next_line_index
            continue

        if _is_markdown_table_content_line(line):
            table_blocks, next_line_index, next_table_index = _consume_markdown_table_group(
                lines,
                line_index,
                block_index,
                table_index + 1,
                heading_path,
            )
            if table_blocks:
                raw_blocks.extend(table_blocks)
                block_index += len(table_blocks)
                table_index = next_table_index
                line_index = next_line_index
                continue

        if MARKDOWN_TABLE_SEPARATOR_PATTERN.match(line):
            line_index += 1
            continue

        unordered_match = MARKDOWN_UNORDERED_LIST_PATTERN.match(line)
        ordered_match = MARKDOWN_ORDERED_LIST_PATTERN.match(line)
        if unordered_match or ordered_match:
            raw_blocks.append(
                _RawBlock(
                    block_type=ExtractedBlockType.LIST_ITEM,
                    raw_text=line,
                    location_base=f"md:l{line_number}",
                    block_index=block_index,
                    heading_path=list(heading_path),
                    start_line=line_number,
                    end_line=line_number,
                    metadata={},
                )
            )
            block_index += 1
            line_index += 1
            continue

        start_line = line_number
        paragraph_lines = [line]
        line_index += 1
        while line_index < len(lines):
            next_line = lines[line_index]
            if (
                not next_line.strip()
                or MARKDOWN_HEADING_PATTERN.match(next_line)
                or _parse_markdown_fence_start(next_line)
                or _is_markdown_table_content_line(next_line)
                or MARKDOWN_TABLE_SEPARATOR_PATTERN.match(next_line)
                or MARKDOWN_UNORDERED_LIST_PATTERN.match(next_line)
                or MARKDOWN_ORDERED_LIST_PATTERN.match(next_line)
            ):
                break
            paragraph_lines.append(next_line)
            line_index += 1

        raw_blocks.append(
            _RawBlock(
                block_type=ExtractedBlockType.PARAGRAPH,
                raw_text="\n".join(paragraph_lines),
                location_base=f"md:p{start_line}",
                block_index=block_index,
                heading_path=list(heading_path),
                start_line=start_line,
                end_line=start_line + len(paragraph_lines) - 1,
                metadata={},
            )
        )
        block_index += 1

    if not raw_blocks:
        return _build_document(
            outcome=ExtractionOutcome.FAILED,
            detected_format=DetectedFileFormat.MD,
            detected_encoding=detected_encoding,
            page_count=None,
            raw_blocks=[],
            warnings=["Markdown 没有可提取文本。"],
            metadata={"line_count": len(lines)},
        )

    return _build_document(
        outcome=ExtractionOutcome.SUCCESS,
        detected_format=DetectedFileFormat.MD,
        detected_encoding=detected_encoding,
        page_count=None,
        raw_blocks=raw_blocks,
        warnings=warnings,
        metadata={"line_count": len(lines)},
    )


def _extract_txt(spool_file: BinaryIO, encoding_hint: str | None) -> ExtractedDocument:
    decoded_text, detected_encoding = _decode_text_bytes(spool_file, encoding_hint)
    if decoded_text is None:
        return _build_document(
            outcome=ExtractionOutcome.FAILED,
            detected_format=DetectedFileFormat.TXT,
            detected_encoding=None,
            page_count=None,
            raw_blocks=[],
            warnings=["TXT 文件编码无法识别。"],
            metadata={},
        )

    lines = _normalize_newlines(decoded_text).split("\n")
    raw_blocks: list[_RawBlock] = []
    block_index = 0
    line_index = 0

    while line_index < len(lines):
        if not lines[line_index].strip():
            line_index += 1
            continue

        start_line = line_index + 1
        paragraph_lines = [lines[line_index]]
        line_index += 1
        while line_index < len(lines) and lines[line_index].strip():
            paragraph_lines.append(lines[line_index])
            line_index += 1

        raw_blocks.append(
            _RawBlock(
                block_type=ExtractedBlockType.PARAGRAPH,
                raw_text="\n".join(paragraph_lines),
                location_base=f"txt:p{block_index}",
                block_index=block_index,
                start_line=start_line,
                end_line=start_line + len(paragraph_lines) - 1,
                metadata={},
            )
        )
        block_index += 1

    if not raw_blocks:
        return _build_document(
            outcome=ExtractionOutcome.FAILED,
            detected_format=DetectedFileFormat.TXT,
            detected_encoding=detected_encoding,
            page_count=None,
            raw_blocks=[],
            warnings=["TXT 没有可提取文本。"],
            metadata={"line_count": len(lines)},
        )

    return _build_document(
        outcome=ExtractionOutcome.SUCCESS,
        detected_format=DetectedFileFormat.TXT,
        detected_encoding=detected_encoding,
        page_count=None,
        raw_blocks=raw_blocks,
        warnings=[],
        metadata={"line_count": len(lines)},
    )


def _decode_text_bytes(spool_file: BinaryIO, encoding_hint: str | None) -> tuple[str | None, str | None]:
    spool_file.seek(0)
    file_bytes = spool_file.read()
    spool_file.seek(0)

    candidates: list[str] = []
    if encoding_hint:
        candidates.append(encoding_hint.strip())

    detected_encoding = _detect_encoding(file_bytes)
    if detected_encoding and detected_encoding not in candidates:
        candidates.append(detected_encoding)

    for encoding in candidates:
        try:
            return file_bytes.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError):
            continue

    return None, None


def _detect_encoding(file_bytes: bytes) -> str | None:
    best_match = from_bytes(file_bytes).best()
    if best_match is None or not best_match.encoding:
        return None
    return best_match.encoding


def _build_document(
    *,
    outcome: ExtractionOutcome,
    detected_format: DetectedFileFormat,
    detected_encoding: str | None,
    page_count: int | None,
    raw_blocks: list[_RawBlock],
    warnings: list[str],
    metadata: dict[str, object],
) -> ExtractedDocument:
    blocks = _materialize_blocks(detected_format, raw_blocks)
    character_count = sum(len(block.normalized_text) for block in blocks)
    return ExtractedDocument(
        outcome=outcome,
        detected_format=detected_format,
        detected_encoding=detected_encoding,
        page_count=page_count,
        character_count=character_count,
        block_count=len(blocks),
        blocks=blocks,
        warnings=warnings,
        metadata=metadata,
    )


def _materialize_blocks(
    detected_format: DetectedFileFormat,
    raw_blocks: list[_RawBlock],
) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    source_order = 0

    for raw_block in raw_blocks:
        raw_segments = _split_long_block(raw_block.raw_text)
        for segment_index, segment_text in enumerate(raw_segments):
            location_key = (
                raw_block.location_base
                if len(raw_segments) == 1
                else f"{raw_block.location_base}:part{segment_index}"
            )
            metadata = dict(raw_block.metadata or {})
            if len(raw_segments) > 1:
                metadata.update(
                    {
                        "segment_index": segment_index,
                        "segment_count": len(raw_segments),
                        "original_location_key": raw_block.location_base,
                    }
                )
            blocks.append(
                ExtractedBlock(
                    source_order=source_order,
                    block_type=raw_block.block_type,
                    raw_text=segment_text,
                    normalized_text=_normalize_block_text(
                        segment_text,
                        collapse_spaces=raw_block.collapse_spaces,
                    ),
                    location_key=location_key,
                    anchor_hash=_build_anchor_hash(detected_format, location_key, segment_text),
                    page_no=raw_block.page_no,
                    block_index=raw_block.block_index,
                    heading_level=raw_block.heading_level,
                    heading_path=list(raw_block.heading_path or []),
                    start_line=raw_block.start_line,
                    end_line=raw_block.end_line,
                    table_index=raw_block.table_index,
                    row_index=raw_block.row_index,
                    metadata=metadata,
                )
            )
            source_order += 1

    return blocks


def _build_anchor_hash(
    detected_format: DetectedFileFormat,
    location_key: str,
    raw_text: str,
) -> str:
    return hashlib.sha256(
        f"{detected_format.value}|{location_key}|{raw_text}".encode("utf-8")
    ).hexdigest()


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_block_text(text: str, *, collapse_spaces: bool) -> str:
    normalized = _normalize_newlines(text)
    if not collapse_spaces:
        return normalized.strip()

    normalized_lines = [
        re.sub(r"[ \t\f\v]+", " ", line).strip()
        for line in normalized.split("\n")
    ]
    return "\n".join(normalized_lines).strip()


def _split_text_blocks(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]


def _split_long_block(text: str) -> list[str]:
    if len(text) <= MAX_BLOCK_CHARACTERS:
        return [text]

    segments: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_BLOCK_CHARACTERS, len(text))
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at <= start:
                split_at = end
        else:
            split_at = end

        segment = text[start:split_at].strip()
        if segment:
            segments.append(segment)
        start = split_at if split_at > start else end
        while start < len(text) and text[start] in {" ", "\n", "\r", "\t"}:
            start += 1

    return segments or [text[:MAX_BLOCK_CHARACTERS]]


def _is_list_paragraph(paragraph: Paragraph, style_name: str) -> bool:
    if style_name.lower().startswith("list"):
        return True
    paragraph_properties = getattr(paragraph._p, "pPr", None)
    return bool(paragraph_properties is not None and paragraph_properties.numPr is not None)


def _is_fence_start(line: str) -> bool:
    return _parse_markdown_fence_start(line) is not None


def _is_markdown_table_content_line(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and not _is_fence_start(stripped)


def _get_docx_style_name(style: BaseStyle | None) -> str:
    return style.name if style is not None and isinstance(style.name, str) else ""


def _get_docx_style_id(style: BaseStyle | None) -> str:
    return style.style_id if style is not None and isinstance(style.style_id, str) else ""


def _get_docx_heading_level(style: BaseStyle | None) -> int | None:
    style_id = _get_docx_style_id(style)
    matched_style_id = DOCX_HEADING_STYLE_ID_PATTERN.fullmatch(style_id)
    if matched_style_id:
        return int(matched_style_id.group(1))

    style_name = _get_docx_style_name(style)
    matched_style_name = DOCX_HEADING_STYLE_NAME_PATTERN.fullmatch(style_name)
    if matched_style_name:
        return int(matched_style_name.group(1))

    return None


def _parse_markdown_fence_start(line: str) -> re.Match[str] | None:
    return MARKDOWN_FENCE_PATTERN.fullmatch(line)


def _consume_markdown_fenced_code_block(
    lines: list[str],
    start_index: int,
    block_index: int,
    code_index: int,
    heading_path: list[str],
    fence_match: re.Match[str],
) -> tuple[_RawBlock, int, str | None]:
    opening_line = lines[start_index]
    start_line = start_index + 1
    fence = fence_match.group("fence")
    fence_char = fence[0]
    fence_length = len(fence)
    info_string = fence_match.group("info").strip()
    code_lines = [opening_line]
    line_index = start_index + 1
    warning: str | None = None
    end_line = start_line

    while line_index < len(lines):
        current_line = lines[line_index]
        code_lines.append(current_line)
        closing_match = MARKDOWN_FENCE_PATTERN.fullmatch(current_line)
        if (
            closing_match is not None
            and closing_match.group("fence")[0] == fence_char
            and len(closing_match.group("fence")) >= fence_length
        ):
            end_line = line_index + 1
            line_index += 1
            break
        line_index += 1
    else:
        end_line = len(lines)
        warning = "Markdown 存在未闭合代码围栏。"

    return (
        _RawBlock(
            block_type=ExtractedBlockType.CODE,
            raw_text="\n".join(code_lines),
            location_base=f"md:c{code_index}:l{start_line}",
            block_index=block_index,
            heading_path=list(heading_path),
            start_line=start_line,
            end_line=end_line,
            metadata={"info_string": info_string},
            collapse_spaces=False,
        ),
        line_index,
        warning,
    )


def _consume_markdown_table_group(
    lines: list[str],
    start_index: int,
    block_index: int,
    table_index: int,
    heading_path: list[str],
) -> tuple[list[_RawBlock], int, int]:
    raw_blocks: list[_RawBlock] = []
    line_index = start_index
    row_index = 0
    current_block_index = block_index

    while line_index < len(lines):
        current_line = lines[line_index]
        if not _is_markdown_table_content_line(current_line) and not MARKDOWN_TABLE_SEPARATOR_PATTERN.match(
            current_line
        ):
            break

        line_number = line_index + 1
        if _is_markdown_table_content_line(current_line) and not MARKDOWN_TABLE_SEPARATOR_PATTERN.match(
            current_line
        ):
            row_index += 1
            cells = [cell.strip() for cell in current_line.strip().strip("|").split("|")]
            raw_blocks.append(
                _RawBlock(
                    block_type=ExtractedBlockType.TABLE_ROW,
                    raw_text=current_line,
                    location_base=f"md:t{table_index}:r{row_index}:l{line_number}",
                    block_index=current_block_index,
                    heading_path=list(heading_path),
                    start_line=line_number,
                    end_line=line_number,
                    table_index=table_index,
                    row_index=row_index,
                    metadata={"cells": cells},
                )
            )
            current_block_index += 1

        line_index += 1

    if not raw_blocks:
        return [], start_index + 1, table_index - 1

    return raw_blocks, line_index, table_index
