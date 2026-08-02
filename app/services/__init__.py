from app.services.dynamic_schema import (
    ArchivedDynamicSchemaError,
    DynamicSchemaActivationTransitionError,
    DynamicSchemaIdentityMismatchError,
    DynamicSchemaNotFoundError,
    DynamicSchemaPermissionError,
    DynamicSchemaProjectNotFoundError,
    DynamicSchemaStateCorruptionError,
    DynamicSchemaServiceError,
    DynamicSchemaVersionNotFoundError,
    activate_dynamic_schema_version,
    create_human_schema_draft,
    propose_ai_schema_version,
)
from app.services.entity import (
    EntityAliasNotFoundError,
    EntityIdentityConflictError,
    EntityNotFoundError,
    EntityPermissionError,
    EntityProjectNotFoundError,
    EntityServiceError,
    EntityStateError,
    PrimaryEntityAliasChangeError,
    PrimaryEntityAliasRetireError,
    add_entity_alias,
    build_entity_identity_hash,
    create_entity_with_primary_alias,
    resolve_entity_alias,
    retire_entity_alias,
)
from app.services.dynamic_schema_projection import (
    DynamicSchemaProjectionError,
    DynamicSchemaProjectionNotFoundError,
    ProjectionStateCorruptionError,
    project_current_dynamic_schema,
    project_dynamic_schema_version,
)
from app.services.dynamic_schema_ufl_projection import (
    DynamicSchemaUFLProjectionError,
    DynamicSchemaUFLProjectionInvariantError,
    DynamicSchemaUFLProjectionStateError,
    authenticate_dynamic_schema_ufl_projected_field,
    authenticate_dynamic_schema_ufl_projection,
    normalize_dynamic_schema_ufl_subject_keys,
    project_orchestration_ufl_to_dynamic_schema,
    serialize_dynamic_schema_ufl_fact,
    serialize_dynamic_schema_ufl_projected_field,
    serialize_dynamic_schema_ufl_projected_record,
)
from app.services.fact import (
    FactIdentityConflictError,
    FactNotFoundError,
    FactPermissionError,
    FactProposalError,
    FactProposalRunNotFoundError,
    FactReferencedEntityMismatchError,
    FactReferencedEntityNotEligibleError,
    FactSubjectEntityConflictError,
    FactSubjectEntityMismatchError,
    FactSubjectEntityNotEligibleError,
    FactValueNotFoundError,
    InvalidFactProposalError,
    NormalizedFactValue,
    RetiredFactError,
    accept_fact_value,
    build_fact_identity_hash,
    create_human_fact_value,
    normalize_fact_value_input,
    propose_ai_fact_value,
    reject_fact_value,
)
from app.services.document_content import (
    DocumentBlockNotFoundError,
    EvidenceOffsetError,
    ExtractionPersistenceError,
    ExtractionRevisionNotFoundError,
    InvalidExtractionResultError,
    create_source_evidence,
    persist_extraction_result,
)
from app.services.document_revision_diff import (
    DocumentRevisionBlockDiffError,
    DocumentRevisionBlockDiffInvariantError,
    DocumentRevisionBlockDiffStateError,
    get_document_revision_block_diff,
)
from app.services.document_revision_fact_diff import (
    DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_NAME,
    DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_VERSION,
    DocumentRevisionFactDiffError,
    DocumentRevisionFactDiffInvariantError,
    DocumentRevisionFactDiffStateError,
    get_document_revision_fact_diff,
)
from app.services.document_extraction import extract_document
from app.services.document_upload import (
    ProjectDocumentSummary,
    ProjectUploadContext,
    RevisionDetailContext,
    UploadAccessError,
    UploadFormError,
    UploadProjectNotFoundError,
    UploadRevisionNotFoundError,
    UploadTransactionResult,
    get_project_upload_context,
    get_project_workspace_context,
    get_revision_detail_for_user,
    upload_document_for_project,
)
from app.services.identity import (
    DuplicateEmailError,
    DuplicateHandleError,
    IdentityConflictError,
    get_active_user_by_id,
    register_user,
)
from app.services.file_ingestion import (
    EmptyUploadError,
    FileIngestionError,
    FileTooLargeError,
    InvalidOriginalFilenameError,
    TechnicalPreflightError,
    get_local_file_storage,
    receive_upload_stream,
    run_technical_preflight,
    sanitize_original_filename,
)
from app.services.project import (
    DuplicateProjectSlugError,
    ProjectConflictError,
    create_project_for_owner,
    get_project_for_user_by_slug,
    list_projects_for_user,
)
from app.services.revision_admission import (
    RevisionAdmissionError,
    RevisionAdmissionNotFoundError,
    RevisionAdmissionPermissionError,
    RevisionAdmissionStateCorruptionError,
    RevisionAdmissionStateError,
    decide_revision_admission,
    resolve_revision_admission,
)
from app.services.revision_extraction import (
    EXTRACTOR_NAME,
    EXTRACTOR_VERSION,
    RevisionExtractionAdmissionStateError,
    RevisionExtractionConfigError,
    RevisionExtractionError,
    RevisionExtractionNotFoundError,
    RevisionExtractionTransitionError,
    run_revision_extraction,
)
from app.services.processing_job import (
    ProcessingJobError,
    ProcessingJobLeaseError,
    ProcessingJobNotFoundError,
    ProcessingJobPermissionError,
    ProcessingJobStateError,
    claim_processing_job,
    complete_processing_job,
    enqueue_revision_extraction_job,
    fail_processing_job,
    recover_stale_revision_extraction,
    renew_processing_job_lease,
    retry_failed_revision_extraction,
)
from app.services.processing_job_executor import execute_revision_extraction_job
from app.services.processing_job_dispatch import dispatch_revision_extraction_job


def __getattr__(name: str):
    if name in {
        "DynamicSchemaReviewProjectionError",
        "DynamicSchemaReviewProjectionStateError",
        "DynamicSchemaReviewProjectionInvariantError",
        "authenticate_dynamic_schema_reviewed_field",
        "authenticate_dynamic_schema_review_projection",
        "project_reviewed_orchestration_ufl_to_dynamic_schema",
        "EffectiveFactValueProjectionError",
        "EffectiveFactValueProjectionStateError",
        "EffectiveFactValueProjectionInvariantError",
        "get_effective_fact_value_projection",
        "authenticate_effective_fact_value_projection",
        "DynamicSchemaKnowledgeViewError",
        "DynamicSchemaKnowledgeViewStateError",
        "DynamicSchemaKnowledgeViewInvariantError",
        "authenticate_dynamic_schema_knowledge_view",
        "build_dynamic_schema_knowledge_view",
        "serialize_dynamic_schema_knowledge_view",
        "ProjectVersionError",
        "ProjectVersionStateError",
        "ProjectVersionInvariantError",
        "authenticate_project_version_snapshot",
        "create_project_version",
        "get_project_version_snapshot",
    }:
        exported = {}
        if name in {
            "DynamicSchemaReviewProjectionError",
            "DynamicSchemaReviewProjectionStateError",
            "DynamicSchemaReviewProjectionInvariantError",
            "authenticate_dynamic_schema_reviewed_field",
            "authenticate_dynamic_schema_review_projection",
            "project_reviewed_orchestration_ufl_to_dynamic_schema",
        }:
            from app.services.dynamic_schema_review_projection import (
                DynamicSchemaReviewProjectionError,
                DynamicSchemaReviewProjectionInvariantError,
                DynamicSchemaReviewProjectionStateError,
                authenticate_dynamic_schema_reviewed_field,
                authenticate_dynamic_schema_review_projection,
                project_reviewed_orchestration_ufl_to_dynamic_schema,
            )

            exported.update(
                {
                    "DynamicSchemaReviewProjectionError": (
                        DynamicSchemaReviewProjectionError
                    ),
                    "DynamicSchemaReviewProjectionStateError": (
                        DynamicSchemaReviewProjectionStateError
                    ),
                    "DynamicSchemaReviewProjectionInvariantError": (
                        DynamicSchemaReviewProjectionInvariantError
                    ),
                    "authenticate_dynamic_schema_reviewed_field": (
                        authenticate_dynamic_schema_reviewed_field
                    ),
                    "authenticate_dynamic_schema_review_projection": (
                        authenticate_dynamic_schema_review_projection
                    ),
                    "project_reviewed_orchestration_ufl_to_dynamic_schema": (
                        project_reviewed_orchestration_ufl_to_dynamic_schema
                    ),
                }
            )
        if name in {
            "EffectiveFactValueProjectionError",
            "EffectiveFactValueProjectionStateError",
            "EffectiveFactValueProjectionInvariantError",
            "get_effective_fact_value_projection",
            "authenticate_effective_fact_value_projection",
        }:
            from app.services.effective_fact_value import (
                EffectiveFactValueProjectionError,
                EffectiveFactValueProjectionInvariantError,
                EffectiveFactValueProjectionStateError,
                authenticate_effective_fact_value_projection,
                get_effective_fact_value_projection,
            )

            exported.update(
                {
                    "EffectiveFactValueProjectionError": (
                        EffectiveFactValueProjectionError
                    ),
                    "EffectiveFactValueProjectionStateError": (
                        EffectiveFactValueProjectionStateError
                    ),
                    "EffectiveFactValueProjectionInvariantError": (
                        EffectiveFactValueProjectionInvariantError
                    ),
                    "get_effective_fact_value_projection": (
                        get_effective_fact_value_projection
                    ),
                    "authenticate_effective_fact_value_projection": (
                        authenticate_effective_fact_value_projection
                    ),
                }
            )
        if name in {
            "DynamicSchemaKnowledgeViewError",
            "DynamicSchemaKnowledgeViewStateError",
            "DynamicSchemaKnowledgeViewInvariantError",
            "authenticate_dynamic_schema_knowledge_view",
            "build_dynamic_schema_knowledge_view",
            "serialize_dynamic_schema_knowledge_view",
        }:
            from app.services.dynamic_schema_knowledge_view import (
                DynamicSchemaKnowledgeViewError,
                DynamicSchemaKnowledgeViewInvariantError,
                DynamicSchemaKnowledgeViewStateError,
                authenticate_dynamic_schema_knowledge_view,
                build_dynamic_schema_knowledge_view,
                serialize_dynamic_schema_knowledge_view,
            )

            exported.update(
                {
                    "DynamicSchemaKnowledgeViewError": (
                        DynamicSchemaKnowledgeViewError
                    ),
                    "DynamicSchemaKnowledgeViewStateError": (
                        DynamicSchemaKnowledgeViewStateError
                    ),
                    "DynamicSchemaKnowledgeViewInvariantError": (
                        DynamicSchemaKnowledgeViewInvariantError
                    ),
                    "authenticate_dynamic_schema_knowledge_view": (
                        authenticate_dynamic_schema_knowledge_view
                    ),
                    "build_dynamic_schema_knowledge_view": (
                        build_dynamic_schema_knowledge_view
                    ),
                    "serialize_dynamic_schema_knowledge_view": (
                        serialize_dynamic_schema_knowledge_view
                    ),
                }
            )
        if name in {
            "ProjectVersionError",
            "ProjectVersionStateError",
            "ProjectVersionInvariantError",
            "authenticate_project_version_snapshot",
            "create_project_version",
            "get_project_version_snapshot",
        }:
            from app.services.project_version import (
                ProjectVersionError,
                ProjectVersionInvariantError,
                ProjectVersionStateError,
                authenticate_project_version_snapshot,
                create_project_version,
                get_project_version_snapshot,
            )

            exported.update(
                {
                    "ProjectVersionError": ProjectVersionError,
                    "ProjectVersionStateError": ProjectVersionStateError,
                    "ProjectVersionInvariantError": ProjectVersionInvariantError,
                    "authenticate_project_version_snapshot": (
                        authenticate_project_version_snapshot
                    ),
                    "create_project_version": create_project_version,
                    "get_project_version_snapshot": get_project_version_snapshot,
                }
            )
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExtractionPersistenceError",
    "ExtractionRevisionNotFoundError",
    "InvalidExtractionResultError",
    "DocumentBlockNotFoundError",
    "EvidenceOffsetError",
    "DocumentRevisionBlockDiffError",
    "DocumentRevisionBlockDiffStateError",
    "DocumentRevisionBlockDiffInvariantError",
    "DocumentRevisionFactDiffError",
    "DocumentRevisionFactDiffStateError",
    "DocumentRevisionFactDiffInvariantError",
    "DynamicSchemaServiceError",
    "DynamicSchemaProjectionError",
    "DynamicSchemaProjectionNotFoundError",
    "ProjectionStateCorruptionError",
    "DynamicSchemaUFLProjectionError",
    "DynamicSchemaUFLProjectionStateError",
    "DynamicSchemaUFLProjectionInvariantError",
    "DynamicSchemaKnowledgeViewError",
    "DynamicSchemaKnowledgeViewStateError",
    "DynamicSchemaKnowledgeViewInvariantError",
    "EffectiveFactValueProjectionError",
    "EffectiveFactValueProjectionStateError",
    "EffectiveFactValueProjectionInvariantError",
    "DynamicSchemaReviewProjectionError",
    "DynamicSchemaReviewProjectionStateError",
    "DynamicSchemaReviewProjectionInvariantError",
    "authenticate_dynamic_schema_reviewed_field",
    "DynamicSchemaPermissionError",
    "DynamicSchemaProjectNotFoundError",
    "DynamicSchemaIdentityMismatchError",
    "DynamicSchemaNotFoundError",
    "DynamicSchemaVersionNotFoundError",
    "DynamicSchemaActivationTransitionError",
    "DynamicSchemaStateCorruptionError",
    "ArchivedDynamicSchemaError",
    "EntityServiceError",
    "EntityPermissionError",
    "EntityProjectNotFoundError",
    "EntityNotFoundError",
    "EntityAliasNotFoundError",
    "EntityIdentityConflictError",
    "EntityStateError",
    "PrimaryEntityAliasChangeError",
    "PrimaryEntityAliasRetireError",
    "build_entity_identity_hash",
    "create_entity_with_primary_alias",
    "add_entity_alias",
    "retire_entity_alias",
    "resolve_entity_alias",
    "activate_dynamic_schema_version",
    "project_dynamic_schema_version",
    "project_current_dynamic_schema",
    "project_orchestration_ufl_to_dynamic_schema",
    "authenticate_dynamic_schema_ufl_projected_field",
    "authenticate_dynamic_schema_ufl_projection",
    "normalize_dynamic_schema_ufl_subject_keys",
    "serialize_dynamic_schema_ufl_fact",
    "serialize_dynamic_schema_ufl_projected_field",
    "serialize_dynamic_schema_ufl_projected_record",
    "get_effective_fact_value_projection",
    "authenticate_effective_fact_value_projection",
    "authenticate_dynamic_schema_reviewed_field",
    "authenticate_dynamic_schema_review_projection",
    "project_reviewed_orchestration_ufl_to_dynamic_schema",
    "authenticate_dynamic_schema_knowledge_view",
    "build_dynamic_schema_knowledge_view",
    "serialize_dynamic_schema_knowledge_view",
    "ProjectVersionError",
    "ProjectVersionStateError",
    "ProjectVersionInvariantError",
    "authenticate_project_version_snapshot",
    "create_project_version",
    "get_project_version_snapshot",
    "create_human_schema_draft",
    "propose_ai_schema_version",
    "FactProposalError",
    "FactProposalRunNotFoundError",
    "FactIdentityConflictError",
    "InvalidFactProposalError",
    "RetiredFactError",
    "FactPermissionError",
    "FactNotFoundError",
    "FactValueNotFoundError",
    "FactSubjectEntityNotEligibleError",
    "FactSubjectEntityMismatchError",
    "FactSubjectEntityConflictError",
    "FactReferencedEntityNotEligibleError",
    "FactReferencedEntityMismatchError",
    "NormalizedFactValue",
    "build_fact_identity_hash",
    "normalize_fact_value_input",
    "propose_ai_fact_value",
    "create_human_fact_value",
    "accept_fact_value",
    "reject_fact_value",
    "persist_extraction_result",
    "create_source_evidence",
    "get_document_revision_block_diff",
    "DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_NAME",
    "DOCUMENT_REVISION_FACT_DIFF_ALGORITHM_VERSION",
    "get_document_revision_fact_diff",
    "extract_document",
    "UploadAccessError",
    "UploadFormError",
    "UploadProjectNotFoundError",
    "UploadRevisionNotFoundError",
    "ProjectUploadContext",
    "ProjectDocumentSummary",
    "RevisionDetailContext",
    "UploadTransactionResult",
    "get_project_upload_context",
    "get_project_workspace_context",
    "get_revision_detail_for_user",
    "upload_document_for_project",
    "FileIngestionError",
    "InvalidOriginalFilenameError",
    "EmptyUploadError",
    "FileTooLargeError",
    "TechnicalPreflightError",
    "sanitize_original_filename",
    "get_local_file_storage",
    "receive_upload_stream",
    "run_technical_preflight",
    "IdentityConflictError",
    "DuplicateHandleError",
    "DuplicateEmailError",
    "get_active_user_by_id",
    "register_user",
    "ProjectConflictError",
    "DuplicateProjectSlugError",
    "list_projects_for_user",
    "get_project_for_user_by_slug",
    "create_project_for_owner",
    "RevisionAdmissionError",
    "RevisionAdmissionNotFoundError",
    "RevisionAdmissionPermissionError",
    "RevisionAdmissionStateError",
    "RevisionAdmissionStateCorruptionError",
    "resolve_revision_admission",
    "decide_revision_admission",
    "RevisionExtractionError",
    "RevisionExtractionNotFoundError",
    "RevisionExtractionAdmissionStateError",
    "RevisionExtractionTransitionError",
    "RevisionExtractionConfigError",
    "EXTRACTOR_NAME",
    "EXTRACTOR_VERSION",
    "run_revision_extraction",
    "ProcessingJobError",
    "ProcessingJobNotFoundError",
    "ProcessingJobPermissionError",
    "ProcessingJobStateError",
    "ProcessingJobLeaseError",
    "enqueue_revision_extraction_job",
    "retry_failed_revision_extraction",
    "recover_stale_revision_extraction",
    "claim_processing_job",
    "complete_processing_job",
    "fail_processing_job",
    "renew_processing_job_lease",
    "execute_revision_extraction_job",
    "dispatch_revision_extraction_job",
]
