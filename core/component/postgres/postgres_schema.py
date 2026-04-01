"""
Configuration de la base cible PostgreSQL et de son schema documentaire.

Tu controles ici :
- le nom exact de la base a creer si elle n'existe pas
- les tables V1/V2 utiles a creer automatiquement si elles manquent
- les index utiles pour les futures requetes
- le schema PostgreSQL dedie `dms`
"""

POSTGRES_DATABASE_NAME = "dms_core"
POSTGRES_EXTENSIONS = []

_SCHEMA_SQL = "CREATE SCHEMA IF NOT EXISTS dms;"

POSTGRES_TABLES = [
    {
        "name": "dms.runs",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.runs (
            run_id TEXT PRIMARY KEY,
            pipeline_profile TEXT NOT NULL,
            profile TEXT NULL,
            pipeline_version TEXT NULL,
            mapping_version TEXT NULL,
            payload_schema_version TEXT NULL,
            source TEXT NULL,
            generated_at TIMESTAMPTZ NULL,
            documents_count INTEGER NULL,
            started_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL,
            status TEXT NULL,
            fusion_path TEXT NULL,
            raw_payload JSONB NULL,
            pipeline_json JSONB NULL,
            postgres_sync_json JSONB NULL,
            null_policy_json JSONB NULL,
            source_context_json JSONB NULL,
            cross_document_analysis_json JSONB NULL,
            registries_json JSONB NULL,
            item_templates_json JSONB NULL,
            sql_mapping_hints_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (documents_count IS NULL OR documents_count >= 0),
            CHECK (status IS NULL OR status IN ('pending', 'running', 'completed', 'failed', 'partial'))
        );
        """,
    },
    {
        "name": "dms.ingest_queue",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.ingest_queue (
            ingest_key TEXT PRIMARY KEY,
            run_id TEXT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            document_id TEXT NULL,
            source_file_path TEXT NULL,
            source_filename TEXT NULL,
            file_sha256 TEXT NULL,
            payload_json JSONB NULL,
            ingest_status TEXT NOT NULL,
            error_message TEXT NULL,
            received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            processed_at TIMESTAMPTZ NULL,
            CHECK (ingest_status IN ('received', 'processing', 'processed', 'failed'))
        );
        """,
    },
    {
        "name": "dms.documents",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.documents (
            document_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            source_document_key TEXT NULL,
            file_sha256 TEXT NULL,
            content_sha256 TEXT NULL,
            document_version INTEGER NULL,
            is_latest BOOLEAN NULL,
            pipeline_profile TEXT NULL,
            source TEXT NULL,
            file_name TEXT NULL,
            file_path_primary TEXT NULL,
            file_paths_json JSONB NULL,
            file_size BIGINT NULL,
            file_page_count INTEGER NULL,
            file_mime TEXT NULL,
            file_ext TEXT NULL,
            file_content_mode TEXT NULL,
            doc_type TEXT NULL,
            classification_doc_id TEXT NULL,
            classification_status TEXT NULL,
            classification_winning_score DOUBLE PRECISION NULL,
            classification_threshold DOUBLE PRECISION NULL,
            classification_margin DOUBLE PRECISION NULL,
            classification_scores_json JSONB NULL,
            classification_keyword_matches_json JSONB NULL,
            classification_log_json JSONB NULL,
            classification_scores_audit_json JSONB NULL,
            anti_confusion_targets_json JSONB NULL,
            classification_decision_debug_json JSONB NULL,
            content_document_kind TEXT NULL,
            content_content_type TEXT NULL,
            content_source TEXT NULL,
            language_primary TEXT NULL,
            languages_json JSONB NULL,
            ingest_status TEXT NULL,
            parse_status TEXT NULL,
            normalization_status TEXT NULL,
            document_quality_score DOUBLE PRECISION NULL,
            ocr_confidence_avg DOUBLE PRECISION NULL,
            table_detection_score_avg DOUBLE PRECISION NULL,
            has_ocr BOOLEAN NULL,
            has_tables BOOLEAN NULL,
            has_visual_marks BOOLEAN NULL,
            has_links BOOLEAN NULL,
            has_errors BOOLEAN NULL,
            last_error_message TEXT NULL,
            warnings_json JSONB NULL,
            logs_json JSONB NULL,
            visual_flags_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (file_size IS NULL OR file_size >= 0),
            CHECK (file_page_count IS NULL OR file_page_count >= 0)
        );
        """,
    },
    {
        "name": "dms.document_texts",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_texts (
            document_id TEXT PRIMARY KEY REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            language TEXT NULL,
            source_title TEXT NULL,
            text_title TEXT NULL,
            text_raw TEXT NULL,
            text_normalized TEXT NULL,
            search_full_text TEXT NULL,
            search_keywords_json JSONB NULL,
            normalization_json JSONB NULL,
            text_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_payloads",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_payloads (
            document_id TEXT PRIMARY KEY REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            payload_schema_version TEXT NULL,
            raw_document_json JSONB NOT NULL,
            source_payload_json JSONB NULL,
            file_json JSONB NULL,
            classification_json JSONB NULL,
            content_json JSONB NULL,
            text_json JSONB NULL,
            document_structure_json JSONB NULL,
            extraction_json JSONB NULL,
            nlp_json JSONB NULL,
            ml50_json JSONB NULL,
            ml100_json JSONB NULL,
            components_json JSONB NULL,
            quality_checks_json JSONB NULL,
            cross_document_json JSONB NULL,
            processing_json JSONB NULL,
            meta_json JSONB NULL,
            ocr_json JSONB NULL,
            human_review_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.run_payload_nodes",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.run_payload_nodes (
            node_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            document_id TEXT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            parent_node_id TEXT NULL REFERENCES dms.run_payload_nodes(node_id) ON DELETE CASCADE,
            json_path TEXT NOT NULL,
            path_hash TEXT NOT NULL,
            key_name TEXT NULL,
            array_index INTEGER NULL,
            node_kind TEXT NOT NULL,
            value_type TEXT NULL,
            value_text TEXT NULL,
            value_number DOUBLE PRECISION NULL,
            value_boolean BOOLEAN NULL,
            value_json JSONB NULL,
            source_section TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (node_kind IN ('object', 'array', 'scalar', 'null'))
        );
        """,
    },
    {
        "name": "dms.document_payload_nodes",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_payload_nodes (
            node_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            parent_node_id TEXT NULL REFERENCES dms.document_payload_nodes(node_id) ON DELETE CASCADE,
            json_path TEXT NOT NULL,
            path_hash TEXT NOT NULL,
            key_name TEXT NULL,
            array_index INTEGER NULL,
            node_kind TEXT NOT NULL,
            value_type TEXT NULL,
            value_text TEXT NULL,
            value_number DOUBLE PRECISION NULL,
            value_boolean BOOLEAN NULL,
            value_json JSONB NULL,
            source_section TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (node_kind IN ('object', 'array', 'scalar', 'null'))
        );
        """,
    },
    {
        "name": "dms.document_identifiers",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_identifiers (
            identifier_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            identifier_type TEXT NULL,
            identifier_scope TEXT NULL,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            value_text TEXT NULL,
            value_normalized TEXT NULL,
            source_component TEXT NULL,
            source_path TEXT NULL,
            confidence_score DOUBLE PRECISION NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_classification_scores",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_classification_scores (
            classification_score_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            doc_type TEXT NOT NULL,
            score DOUBLE PRECISION NULL,
            score_rank INTEGER NULL,
            is_winner BOOLEAN NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_classification_score_audit",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_classification_score_audit (
            score_audit_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            doc_type TEXT NULL,
            metric_name TEXT NULL,
            metric_value_text TEXT NULL,
            metric_value_number DOUBLE PRECISION NULL,
            metric_value_boolean BOOLEAN NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_classification_keyword_matches",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_classification_keyword_matches (
            keyword_match_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            bucket_name TEXT NULL,
            doc_type TEXT NULL,
            keyword_text TEXT NULL,
            match_count INTEGER NULL,
            weight DOUBLE PRECISION NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_anti_confusion_hits",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_anti_confusion_hits (
            anti_confusion_hit_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            target_doc_type TEXT NULL,
            keyword_text TEXT NULL,
            hit_count INTEGER NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_anti_confusion_targets",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_anti_confusion_targets (
            anti_confusion_target_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            target_doc_type TEXT NOT NULL,
            target_rank INTEGER NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_extraction_summaries",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_extraction_summaries (
            document_id TEXT PRIMARY KEY REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            extraction_method TEXT NULL,
            regex_matches_count INTEGER NULL,
            business_keys_count INTEGER NULL,
            relations_count INTEGER NULL,
            bm25_hits_count INTEGER NULL,
            tables_count INTEGER NULL,
            table_rows_total INTEGER NULL,
            totals_verification_status TEXT NULL,
            totals_verification_passed BOOLEAN NULL,
            totals_verification_complete BOOLEAN NULL,
            visual_detections_count INTEGER NULL,
            native_json JSONB NULL,
            tesseract_json JSONB NULL,
            regex_extractions_json JSONB NULL,
            business_json JSONB NULL,
            relations_json JSONB NULL,
            bm25_json JSONB NULL,
            table_extraction_json JSONB NULL,
            totals_verification_json JSONB NULL,
            visual_detection_json JSONB NULL,
            extraction_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_extractions",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_extractions (
            extraction_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            field_name TEXT NOT NULL,
            field_type TEXT NULL,
            rule_id TEXT NULL,
            is_many BOOLEAN NULL,
            value_text TEXT NULL,
            value_json JSONB NULL,
            source_component TEXT NULL,
            source_path TEXT NULL,
            extractor_name TEXT NULL,
            extractor_version TEXT NULL,
            confidence_score DOUBLE PRECISION NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_regex_fields",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_regex_fields (
            regex_field_id TEXT PRIMARY KEY,
            extraction_id TEXT NULL REFERENCES dms.document_extractions(extraction_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            field_name TEXT NOT NULL,
            field_type TEXT NULL,
            rule_id TEXT NULL,
            is_many BOOLEAN NULL,
            values_count INTEGER NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_regex_matches",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_regex_matches (
            regex_match_id TEXT PRIMARY KEY,
            regex_field_id TEXT NULL REFERENCES dms.document_regex_fields(regex_field_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            field_name TEXT NOT NULL,
            page_index INTEGER NULL,
            start_offset INTEGER NULL,
            end_offset INTEGER NULL,
            match_value TEXT NULL,
            snippet TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_bm25_chunks",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_bm25_chunks (
            bm25_chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            score DOUBLE PRECISION NULL,
            text_preview TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_relations",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_relations (
            relation_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            relation_type TEXT NULL,
            subject_text TEXT NULL,
            predicate_text TEXT NULL,
            object_text TEXT NULL,
            evidence_text TEXT NULL,
            source_path TEXT NULL,
            confidence_score DOUBLE PRECISION NULL,
            source_location_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_tables",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_tables (
            document_table_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            source_block TEXT NOT NULL,
            table_index INTEGER NULL,
            page_index INTEGER NULL,
            table_type TEXT NULL,
            header_map_json JSONB NULL,
            header_score DOUBLE PRECISION NULL,
            rows_count INTEGER NULL,
            detected_columns_json JSONB NULL,
            totals_json JSONB NULL,
            shape_json JSONB NULL,
            raw_table_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_table_rows",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_table_rows (
            table_row_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            document_table_id TEXT NULL REFERENCES dms.document_tables(document_table_id) ON DELETE CASCADE,
            source_block TEXT NOT NULL,
            table_index INTEGER NULL,
            page_index INTEGER NULL,
            row_index INTEGER NULL,
            reference TEXT NULL,
            product TEXT NULL,
            description TEXT NULL,
            quantity DOUBLE PRECISION NULL,
            unit_price DOUBLE PRECISION NULL,
            total_ht DOUBLE PRECISION NULL,
            total_ttc DOUBLE PRECISION NULL,
            total DOUBLE PRECISION NULL,
            computed_total DOUBLE PRECISION NULL,
            effective_total DOUBLE PRECISION NULL,
            effective_total_source TEXT NULL,
            difference DOUBLE PRECISION NULL,
            status TEXT NULL,
            confidence DOUBLE PRECISION NULL,
            raw_cells_json JSONB NULL,
            raw_row_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_table_cells",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_table_cells (
            table_cell_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            document_table_id TEXT NULL REFERENCES dms.document_tables(document_table_id) ON DELETE CASCADE,
            table_row_id TEXT NULL REFERENCES dms.document_table_rows(table_row_id) ON DELETE CASCADE,
            source_block TEXT NOT NULL,
            table_index INTEGER NULL,
            page_index INTEGER NULL,
            row_index INTEGER NULL,
            col_index INTEGER NULL,
            column_name TEXT NULL,
            header_path TEXT NULL,
            cell_role TEXT NULL,
            raw_text TEXT NULL,
            normalized_text TEXT NULL,
            numeric_value DOUBLE PRECISION NULL,
            value_type TEXT NULL,
            unit_text TEXT NULL,
            currency_text TEXT NULL,
            confidence DOUBLE PRECISION NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_quality_checks",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_quality_checks (
            quality_check_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            source_block TEXT NOT NULL,
            check_name TEXT NULL,
            engine TEXT NULL,
            status TEXT NULL,
            passed BOOLEAN NULL,
            complete BOOLEAN NULL,
            rows_total INTEGER NULL,
            row_ok_count INTEGER NULL,
            row_partial_count INTEGER NULL,
            row_mismatch_count INTEGER NULL,
            rows_verified INTEGER NULL,
            computed_subtotal DOUBLE PRECISION NULL,
            declared_subtotal DOUBLE PRECISION NULL,
            declared_tax DOUBLE PRECISION NULL,
            computed_tax DOUBLE PRECISION NULL,
            declared_total DOUBLE PRECISION NULL,
            expected_total DOUBLE PRECISION NULL,
            subtotal_status TEXT NULL,
            tax_status TEXT NULL,
            total_status TEXT NULL,
            declared_totals_raw_json JSONB NULL,
            tolerance TEXT NULL,
            table_anchor_json JSONB NULL,
            subtotal_location_json JSONB NULL,
            tax_location_json JSONB NULL,
            total_location_json JSONB NULL,
            issue_locations_json JSONB NULL,
            details_json JSONB NULL,
            raw_check_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_quality_issue_locations",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_quality_issue_locations (
            issue_id TEXT PRIMARY KEY,
            quality_check_id TEXT NOT NULL REFERENCES dms.document_quality_checks(quality_check_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            kind TEXT NULL,
            table_index INTEGER NULL,
            page_index INTEGER NULL,
            row_index INTEGER NULL,
            computed DOUBLE PRECISION NULL,
            declared DOUBLE PRECISION NULL,
            source_location_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_quality_row_audit",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_quality_row_audit (
            row_audit_id TEXT PRIMARY KEY,
            quality_check_id TEXT NOT NULL REFERENCES dms.document_quality_checks(quality_check_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            table_index INTEGER NULL,
            page_index INTEGER NULL,
            row_index INTEGER NULL,
            quantity DOUBLE PRECISION NULL,
            unit_price DOUBLE PRECISION NULL,
            computed_total DOUBLE PRECISION NULL,
            effective_total DOUBLE PRECISION NULL,
            effective_total_source TEXT NULL,
            difference DOUBLE PRECISION NULL,
            status TEXT NULL,
            source_location_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_quality_declared_locations",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_quality_declared_locations (
            declared_location_id TEXT PRIMARY KEY,
            quality_check_id TEXT NOT NULL REFERENCES dms.document_quality_checks(quality_check_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            location_kind TEXT NULL,
            page_index INTEGER NULL,
            table_index INTEGER NULL,
            source_location_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_quality_check_steps",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_quality_check_steps (
            check_step_id TEXT PRIMARY KEY,
            quality_check_id TEXT NOT NULL REFERENCES dms.document_quality_checks(quality_check_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            step_index INTEGER NULL,
            step_name TEXT NULL,
            status TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_links",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_links (
            link_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            source_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            target_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            source_filename TEXT NULL,
            target_filename TEXT NULL,
            link_type TEXT NULL,
            score DOUBLE PRECISION NULL,
            vector_profile TEXT NULL,
            embedding_method TEXT NULL,
            embedding_backend TEXT NULL,
            vector_dim INTEGER NULL,
            doc_similarity DOUBLE PRECISION NULL,
            sentence_matches_count INTEGER NULL,
            chunk_matches_count INTEGER NULL,
            shared_topics_json JSONB NULL,
            shared_terms_json JSONB NULL,
            score_breakdown_json JSONB NULL,
            audit_json JSONB NULL,
            vector_audit_json JSONB NULL,
            raw_link_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (source_document_id <> target_document_id)
        );
        """,
    },
    {
        "name": "dms.document_link_shared_terms",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_link_shared_terms (
            link_shared_term_id TEXT PRIMARY KEY,
            link_id TEXT NOT NULL REFERENCES dms.document_links(link_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            source_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            target_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            term_rank INTEGER NULL,
            term TEXT NULL,
            score DOUBLE PRECISION NULL,
            doc_a_examples_json JSONB NULL,
            doc_b_examples_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_link_shared_topics",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_link_shared_topics (
            link_shared_topic_id TEXT PRIMARY KEY,
            link_id TEXT NOT NULL REFERENCES dms.document_links(link_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            source_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            target_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            topic_rank INTEGER NULL,
            term TEXT NULL,
            score DOUBLE PRECISION NULL,
            doc_a_examples_json JSONB NULL,
            doc_b_examples_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_link_sentence_matches",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_link_sentence_matches (
            link_sentence_match_id TEXT PRIMARY KEY,
            link_id TEXT NOT NULL REFERENCES dms.document_links(link_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            source_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            target_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            match_index INTEGER NULL,
            score DOUBLE PRECISION NULL,
            shared_terms_json JSONB NULL,
            shared_topics_json JSONB NULL,
            phrase_a_json JSONB NULL,
            phrase_b_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_link_chunk_matches",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_link_chunk_matches (
            link_chunk_match_id TEXT PRIMARY KEY,
            link_id TEXT NOT NULL REFERENCES dms.document_links(link_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            source_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            target_document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            match_index INTEGER NULL,
            score DOUBLE PRECISION NULL,
            vector_similarity DOUBLE PRECISION NULL,
            shared_terms_json JSONB NULL,
            shared_topics_json JSONB NULL,
            chunk_a_json JSONB NULL,
            chunk_b_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_pipeline_features",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_pipeline_features (
            feature_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            pipeline_profile TEXT NULL,
            feature_group TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            backend TEXT NULL,
            method TEXT NULL,
            model TEXT NULL,
            vector_dim INTEGER NULL,
            count_value INTEGER NULL,
            score_value DOUBLE PRECISION NULL,
            text_value TEXT NULL,
            bool_value BOOLEAN NULL,
            topics_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_component_audit",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_component_audit (
            component_audit_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            component_name TEXT NOT NULL,
            backend TEXT NULL,
            method TEXT NULL,
            model TEXT NULL,
            status_text TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_pages",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_pages (
            page_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            lang TEXT NULL,
            chars INTEGER NULL,
            source_path TEXT NULL,
            page_text TEXT NULL,
            text_excerpt TEXT NULL,
            raw_page_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_pages_meta",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_pages_meta (
            page_meta_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_id TEXT NULL REFERENCES dms.document_pages(page_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            width DOUBLE PRECISION NULL,
            height DOUBLE PRECISION NULL,
            rotation DOUBLE PRECISION NULL,
            dpi DOUBLE PRECISION NULL,
            lang TEXT NULL,
            source_path TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_sections",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_sections (
            section_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_blocks",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_blocks (
            block_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_lines",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_lines (
            line_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_words",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_words (
            word_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_headers",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_headers (
            header_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_footers",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_footers (
            footer_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_lists",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_lists (
            list_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_figures",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_figures (
            figure_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_equations",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_equations (
            equation_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_key_value_pairs",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_key_value_pairs (
            key_value_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_reading_order",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_reading_order (
            reading_order_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_non_text_regions",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_non_text_regions (
            non_text_region_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            item_index INTEGER NULL,
            text_excerpt TEXT NULL,
            raw_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_visual_marks",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_visual_marks (
            visual_mark_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            mark_type TEXT NULL,
            kind TEXT NULL,
            score DOUBLE PRECISION NULL,
            confidence DOUBLE PRECISION NULL,
            source TEXT NULL,
            engine TEXT NULL,
            decoded_value TEXT NULL,
            page_width INTEGER NULL,
            page_height INTEGER NULL,
            bbox_px_json JSONB NULL,
            bbox_norm_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_text_normalization_items",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_text_normalization_items (
            normalization_item_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            item_index INTEGER NULL,
            field_name TEXT NULL,
            lang TEXT NULL,
            original_text TEXT NULL,
            normalized_text TEXT NULL,
            method TEXT NULL,
            source_path TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_search_keywords",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_search_keywords (
            search_keyword_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            keyword_rank INTEGER NULL,
            keyword_text TEXT NULL,
            score DOUBLE PRECISION NULL,
            source_path TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_business_fields",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_business_fields (
            business_field_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            field_group TEXT NULL,
            field_name TEXT NOT NULL,
            source_component TEXT NULL,
            source_path TEXT NULL,
            value_text TEXT NULL,
            value_number DOUBLE PRECISION NULL,
            value_boolean BOOLEAN NULL,
            value_json JSONB NULL,
            confidence_score DOUBLE PRECISION NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_topics",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_topics (
            topic_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            pipeline_profile TEXT NULL,
            topic_scope TEXT NULL,
            topic_source TEXT NULL,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            topic_rank INTEGER NULL,
            is_primary BOOLEAN NULL,
            term TEXT NULL,
            score DOUBLE PRECISION NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_sentences",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_sentences (
            sentence_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            char_start INTEGER NULL,
            char_end INTEGER NULL,
            lang TEXT NULL,
            text TEXT NULL,
            text_normalized TEXT NULL,
            tokens_count INTEGER NULL,
            source_location_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_sentence_layouts",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_sentence_layouts (
            sentence_layout_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            sentence_id TEXT NULL REFERENCES dms.document_sentences(sentence_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            line INTEGER NULL,
            col DOUBLE PRECISION NULL,
            col_index INTEGER NULL,
            layout_kind TEXT NULL,
            is_sentence BOOLEAN NULL,
            is_noise BOOLEAN NULL,
            nonspace INTEGER NULL,
            source_path TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_sentence_spans",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_sentence_spans (
            sentence_span_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            sentence_id TEXT NULL REFERENCES dms.document_sentences(sentence_id) ON DELETE CASCADE,
            sentence_layout_id TEXT NULL REFERENCES dms.document_sentence_layouts(sentence_layout_id) ON DELETE CASCADE,
            span_index INTEGER NULL,
            page_index INTEGER NULL,
            line INTEGER NULL,
            col DOUBLE PRECISION NULL,
            col_index INTEGER NULL,
            start_offset INTEGER NULL,
            end_offset INTEGER NULL,
            text TEXT NULL,
            bbox_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_tokens",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_tokens (
            token_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            tok_index INTEGER NULL,
            char_start INTEGER NULL,
            char_end INTEGER NULL,
            lang TEXT NULL,
            token TEXT NULL,
            lemma TEXT NULL,
            pos TEXT NULL,
            ner TEXT NULL,
            xlmr_backend TEXT NULL,
            source_location_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_nlp_matches",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_nlp_matches (
            nlp_match_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            char_start INTEGER NULL,
            char_end INTEGER NULL,
            match_kind TEXT NULL,
            score DOUBLE PRECISION NULL,
            text_excerpt TEXT NULL,
            source_location_json JSONB NULL,
            shared_terms_json JSONB NULL,
            shared_topics_json JSONB NULL,
            phrase_a_json JSONB NULL,
            phrase_b_json JSONB NULL,
            chunk_a_json JSONB NULL,
            chunk_b_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_entities",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_entities (
            entity_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            char_start INTEGER NULL,
            char_end INTEGER NULL,
            lang TEXT NULL,
            entity_type TEXT NULL,
            text TEXT NULL,
            text_normalized TEXT NULL,
            canonical_text TEXT NULL,
            source_location_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_vectors",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_vectors (
            vector_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            pipeline_profile TEXT NULL,
            vector_scope TEXT NULL,
            method TEXT NULL,
            model TEXT NULL,
            vector_dim INTEGER NULL,
            vector_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_chunk_embeddings",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_chunk_embeddings (
            chunk_embedding_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            pipeline_profile TEXT NULL,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            lang TEXT NULL,
            token_count INTEGER NULL,
            text_preview TEXT NULL,
            chunk_primary_topic TEXT NULL,
            chunk_topics_json JSONB NULL,
            vector_dim INTEGER NULL,
            vector_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_word_embeddings",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_word_embeddings (
            word_embedding_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            pipeline_profile TEXT NULL,
            page_index INTEGER NULL,
            sent_index INTEGER NULL,
            tok_index INTEGER NULL,
            lang TEXT NULL,
            token TEXT NULL,
            lemma TEXT NULL,
            vector_dim INTEGER NULL,
            vector_json JSONB NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_layout_header_rows",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_layout_header_rows (
            header_row_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            table_index INTEGER NULL,
            row_index INTEGER NULL,
            text TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_layout_header_cells",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_layout_header_cells (
            header_cell_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            header_row_id TEXT NULL REFERENCES dms.document_layout_header_rows(header_row_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            table_index INTEGER NULL,
            row_index INTEGER NULL,
            col_index INTEGER NULL,
            text TEXT NULL,
            normalized_text TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_layout_table_rows",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_layout_table_rows (
            layout_table_row_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            document_table_id TEXT NULL REFERENCES dms.document_tables(document_table_id) ON DELETE CASCADE,
            page_index INTEGER NULL,
            table_index INTEGER NULL,
            row_index INTEGER NULL,
            line INTEGER NULL,
            col DOUBLE PRECISION NULL,
            col_index INTEGER NULL,
            layout_kind TEXT NULL,
            is_sentence BOOLEAN NULL,
            is_noise BOOLEAN NULL,
            nonspace INTEGER NULL,
            text TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_human_review_tasks",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_human_review_tasks (
            human_review_task_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            task_index INTEGER NULL,
            task_type TEXT NULL,
            status TEXT NULL,
            title TEXT NULL,
            assignee TEXT NULL,
            due_at TIMESTAMPTZ NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_processing_warnings",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_processing_warnings (
            processing_warning_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            warning_index INTEGER NULL,
            level TEXT NULL,
            code TEXT NULL,
            message TEXT NULL,
            source_component TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_processing_logs",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_processing_logs (
            processing_log_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            log_index INTEGER NULL,
            level TEXT NULL,
            message TEXT NULL,
            source_component TEXT NULL,
            logged_at TIMESTAMPTZ NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_processing_steps",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_processing_steps (
            processing_step_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            step_index INTEGER NULL,
            step_name TEXT NULL,
            source_component TEXT NULL,
            status TEXT NULL,
            started_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL,
            duration_ms DOUBLE PRECISION NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_processing_durations",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_processing_durations (
            processing_duration_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            metric_name TEXT NOT NULL,
            source_component TEXT NULL,
            duration_ms DOUBLE PRECISION NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.document_component_metrics",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.document_component_metrics (
            component_metric_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES dms.documents(document_id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES dms.runs(run_id) ON DELETE CASCADE,
            component_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value_text TEXT NULL,
            metric_value_number DOUBLE PRECISION NULL,
            metric_value_boolean BOOLEAN NULL,
            metric_value_json JSONB NULL,
            source_path TEXT NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
    {
        "name": "dms.stable_id_registry",
        "sql": f"""
        CREATE TABLE IF NOT EXISTS dms.stable_id_registry (
            registry_id TEXT PRIMARY KEY,
            entity_kind TEXT NOT NULL,
            natural_key_hash TEXT NOT NULL,
            stable_id TEXT NOT NULL,
            first_seen_run_id TEXT NULL REFERENCES dms.runs(run_id) ON DELETE SET NULL,
            last_seen_run_id TEXT NULL REFERENCES dms.runs(run_id) ON DELETE SET NULL,
            payload_json JSONB NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
    },
]

POSTGRES_INDEXES = [
    {"name": "idx_dms_runs_completed_at", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_runs_completed_at ON dms.runs (completed_at DESC);"},
    {"name": "idx_dms_runs_profile", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_runs_profile ON dms.runs (pipeline_profile);"},
    {"name": "idx_dms_ingest_queue_run", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_ingest_queue_run ON dms.ingest_queue (run_id, ingest_status);"},
    {"name": "idx_dms_documents_run", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_documents_run ON dms.documents (run_id);"},
    {"name": "idx_dms_documents_type", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_documents_type ON dms.documents (doc_type, classification_status);"},
    {"name": "idx_dms_documents_filename", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_documents_filename ON dms.documents (file_name);"},
    {"name": "idx_dms_documents_source_key", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_documents_source_key ON dms.documents (source_document_key);"},
    {"name": "idx_dms_documents_file_sha256", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_documents_file_sha256 ON dms.documents (file_sha256);"},
    {"name": "idx_dms_run_payload_nodes_run_path", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_run_payload_nodes_run_path ON dms.run_payload_nodes (run_id, json_path);"},
    {"name": "idx_dms_run_payload_nodes_path_hash", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_run_payload_nodes_path_hash ON dms.run_payload_nodes (path_hash);"},
    {"name": "idx_dms_run_payload_nodes_section", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_run_payload_nodes_section ON dms.run_payload_nodes (source_section);"},
    {"name": "idx_dms_document_payload_nodes_doc_path", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_payload_nodes_doc_path ON dms.document_payload_nodes (document_id, json_path);"},
    {"name": "idx_dms_document_payload_nodes_path_hash", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_payload_nodes_path_hash ON dms.document_payload_nodes (path_hash);"},
    {"name": "idx_dms_document_payload_nodes_section", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_payload_nodes_section ON dms.document_payload_nodes (source_section);"},
    {"name": "idx_dms_document_identifiers_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_identifiers_doc ON dms.document_identifiers (document_id, identifier_type, value_normalized);"},
    {"name": "idx_dms_classification_scores_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_classification_scores_doc ON dms.document_classification_scores (document_id, doc_type, score_rank);"},
    {"name": "idx_dms_classification_score_audit_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_classification_score_audit_doc ON dms.document_classification_score_audit (document_id, doc_type, metric_name);"},
    {"name": "idx_dms_classification_keyword_matches_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_classification_keyword_matches_doc ON dms.document_classification_keyword_matches (document_id, bucket_name, doc_type);"},
    {"name": "idx_dms_anti_confusion_hits_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_anti_confusion_hits_doc ON dms.document_anti_confusion_hits (document_id, target_doc_type);"},
    {"name": "idx_dms_anti_confusion_targets_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_anti_confusion_targets_doc ON dms.document_anti_confusion_targets (document_id, target_doc_type);"},
    {"name": "idx_dms_document_extractions_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_extractions_doc ON dms.document_extractions (document_id, field_name);"},
    {"name": "idx_dms_document_tables_doc_page", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_tables_doc_page ON dms.document_tables (document_id, page_index, table_index);"},
    {"name": "idx_dms_document_table_rows_doc_table", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_table_rows_doc_table ON dms.document_table_rows (document_id, document_table_id, row_index);"},
    {"name": "idx_dms_document_table_cells_doc_row", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_table_cells_doc_row ON dms.document_table_cells (document_id, document_table_id, row_index, col_index);"},
    {"name": "idx_dms_document_quality_checks_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_quality_checks_doc ON dms.document_quality_checks (document_id, check_name, status);"},
    {"name": "idx_dms_document_quality_issue_locations_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_quality_issue_locations_doc ON dms.document_quality_issue_locations (document_id, quality_check_id, page_index, table_index);"},
    {"name": "idx_dms_document_quality_row_audit_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_quality_row_audit_doc ON dms.document_quality_row_audit (document_id, quality_check_id, page_index, table_index, row_index);"},
    {"name": "idx_dms_document_links_run", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_links_run ON dms.document_links (run_id, source_document_id, target_document_id);"},
    {"name": "uq_dms_document_links_pair", "sql": "CREATE UNIQUE INDEX IF NOT EXISTS uq_dms_document_links_pair ON dms.document_links (run_id, source_document_id, target_document_id, link_type);"},
    {"name": "idx_dms_document_link_shared_terms_link", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_link_shared_terms_link ON dms.document_link_shared_terms (link_id, term_rank);"},
    {"name": "idx_dms_document_link_shared_topics_link", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_link_shared_topics_link ON dms.document_link_shared_topics (link_id, topic_rank);"},
    {"name": "idx_dms_document_link_sentence_matches_link", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_link_sentence_matches_link ON dms.document_link_sentence_matches (link_id, match_index);"},
    {"name": "idx_dms_document_link_chunk_matches_link", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_link_chunk_matches_link ON dms.document_link_chunk_matches (link_id, match_index);"},
    {"name": "idx_dms_document_pipeline_features_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_pipeline_features_doc ON dms.document_pipeline_features (document_id, feature_group, feature_name);"},
    {"name": "idx_dms_document_component_audit_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_component_audit_doc ON dms.document_component_audit (document_id, component_name);"},
    {"name": "idx_dms_document_pages_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_pages_doc ON dms.document_pages (document_id, page_index);"},
    {"name": "idx_dms_document_pages_meta_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_pages_meta_doc ON dms.document_pages_meta (document_id, page_index);"},
    {"name": "idx_dms_document_text_normalization_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_text_normalization_doc ON dms.document_text_normalization_items (document_id, item_index);"},
    {"name": "idx_dms_document_search_keywords_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_search_keywords_doc ON dms.document_search_keywords (document_id, keyword_rank);"},
    {"name": "idx_dms_document_business_fields_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_business_fields_doc ON dms.document_business_fields (document_id, field_group, field_name);"},
    {"name": "idx_dms_document_sentences_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_sentences_doc ON dms.document_sentences (document_id, page_index, sent_index);"},
    {"name": "idx_dms_document_sentence_layouts_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_sentence_layouts_doc ON dms.document_sentence_layouts (document_id, page_index, sent_index);"},
    {"name": "idx_dms_document_sentence_spans_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_sentence_spans_doc ON dms.document_sentence_spans (document_id, page_index, span_index);"},
    {"name": "idx_dms_document_tokens_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_tokens_doc ON dms.document_tokens (document_id, page_index, sent_index, tok_index);"},
    {"name": "idx_dms_document_nlp_matches_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_nlp_matches_doc ON dms.document_nlp_matches (document_id, page_index, sent_index, match_kind);"},
    {"name": "idx_dms_document_entities_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_entities_doc ON dms.document_entities (document_id, entity_type);"},
    {"name": "idx_dms_document_visual_marks_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_visual_marks_doc ON dms.document_visual_marks (document_id, page_index, mark_type);"},
    {"name": "idx_dms_document_layout_header_rows_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_layout_header_rows_doc ON dms.document_layout_header_rows (document_id, page_index, table_index, row_index);"},
    {"name": "idx_dms_document_layout_header_cells_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_layout_header_cells_doc ON dms.document_layout_header_cells (document_id, page_index, table_index, row_index, col_index);"},
    {"name": "idx_dms_document_layout_table_rows_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_layout_table_rows_doc ON dms.document_layout_table_rows (document_id, page_index, table_index, row_index);"},
    {"name": "idx_dms_document_regex_fields_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_regex_fields_doc ON dms.document_regex_fields (document_id, field_name);"},
    {"name": "idx_dms_document_regex_matches_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_regex_matches_doc ON dms.document_regex_matches (document_id, field_name, page_index);"},
    {"name": "idx_dms_document_bm25_chunks_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_bm25_chunks_doc ON dms.document_bm25_chunks (document_id, page_index, sent_index);"},
    {"name": "idx_dms_document_relations_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_relations_doc ON dms.document_relations (document_id, relation_type);"},
    {"name": "idx_dms_document_topics_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_topics_doc ON dms.document_topics (document_id, pipeline_profile, topic_scope);"},
    {"name": "idx_dms_document_vectors_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_vectors_doc ON dms.document_vectors (document_id, pipeline_profile, vector_scope);"},
    {"name": "idx_dms_document_chunk_embeddings_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_chunk_embeddings_doc ON dms.document_chunk_embeddings (document_id, pipeline_profile, page_index, sent_index);"},
    {"name": "idx_dms_document_word_embeddings_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_word_embeddings_doc ON dms.document_word_embeddings (document_id, pipeline_profile, page_index, sent_index, tok_index);"},
    {"name": "idx_dms_document_human_review_tasks_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_human_review_tasks_doc ON dms.document_human_review_tasks (document_id, task_index, status);"},
    {"name": "idx_dms_document_processing_warnings_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_processing_warnings_doc ON dms.document_processing_warnings (document_id, warning_index, level);"},
    {"name": "idx_dms_document_processing_logs_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_processing_logs_doc ON dms.document_processing_logs (document_id, log_index);"},
    {"name": "idx_dms_document_processing_steps_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_processing_steps_doc ON dms.document_processing_steps (document_id, step_index, step_name);"},
    {"name": "idx_dms_document_processing_durations_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_processing_durations_doc ON dms.document_processing_durations (document_id, metric_name);"},
    {"name": "idx_dms_document_component_metrics_doc", "sql": "CREATE INDEX IF NOT EXISTS idx_dms_document_component_metrics_doc ON dms.document_component_metrics (document_id, component_name, metric_name);"},
    {"name": "uq_dms_stable_id_registry_key", "sql": "CREATE UNIQUE INDEX IF NOT EXISTS uq_dms_stable_id_registry_key ON dms.stable_id_registry (entity_kind, natural_key_hash);"},
    {"name": "uq_dms_stable_id_registry_stable_id", "sql": "CREATE UNIQUE INDEX IF NOT EXISTS uq_dms_stable_id_registry_stable_id ON dms.stable_id_registry (stable_id);"},
]

POSTGRES_POST_SQL = [
    _SCHEMA_SQL,
    """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'dms' AND table_name = 'document_extractions'
        ) AND NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'dms' AND table_name = 'document_extractions' AND column_name = 'extraction_id'
        ) THEN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'dms' AND table_name = 'document_extraction_summaries_legacy'
            ) THEN
                ALTER TABLE dms.document_extractions RENAME TO document_extraction_summaries_legacy;
            END IF;
        END IF;
    END $$;
    """,
    "ALTER TABLE IF EXISTS dms.document_relations ADD COLUMN IF NOT EXISTS subject_text TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_relations ADD COLUMN IF NOT EXISTS predicate_text TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_relations ADD COLUMN IF NOT EXISTS object_text TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_relations ADD COLUMN IF NOT EXISTS evidence_text TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_relations ADD COLUMN IF NOT EXISTS source_location_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS rows_total INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS row_ok_count INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS row_partial_count INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS computed_tax DOUBLE PRECISION NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS subtotal_status TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS tax_status TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS total_status TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS declared_totals_raw_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS tolerance TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS table_anchor_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS subtotal_location_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS tax_location_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_quality_checks ADD COLUMN IF NOT EXISTS total_location_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_visual_marks ADD COLUMN IF NOT EXISTS kind TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_visual_marks ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NULL;",
    "ALTER TABLE IF EXISTS dms.document_visual_marks ADD COLUMN IF NOT EXISTS engine TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_visual_marks ADD COLUMN IF NOT EXISTS decoded_value TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_visual_marks ADD COLUMN IF NOT EXISTS page_width INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_visual_marks ADD COLUMN IF NOT EXISTS page_height INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_sentences ADD COLUMN IF NOT EXISTS char_start INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_sentences ADD COLUMN IF NOT EXISTS char_end INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_sentences ADD COLUMN IF NOT EXISTS text_normalized TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_sentences ADD COLUMN IF NOT EXISTS tokens_count INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_sentences ADD COLUMN IF NOT EXISTS source_location_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_tokens ADD COLUMN IF NOT EXISTS char_start INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_tokens ADD COLUMN IF NOT EXISTS char_end INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_tokens ADD COLUMN IF NOT EXISTS xlmr_backend TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_tokens ADD COLUMN IF NOT EXISTS source_location_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_nlp_matches ADD COLUMN IF NOT EXISTS char_start INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_nlp_matches ADD COLUMN IF NOT EXISTS char_end INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_nlp_matches ADD COLUMN IF NOT EXISTS source_location_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_nlp_matches ADD COLUMN IF NOT EXISTS shared_terms_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_nlp_matches ADD COLUMN IF NOT EXISTS shared_topics_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_nlp_matches ADD COLUMN IF NOT EXISTS phrase_a_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_nlp_matches ADD COLUMN IF NOT EXISTS phrase_b_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_nlp_matches ADD COLUMN IF NOT EXISTS chunk_a_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_nlp_matches ADD COLUMN IF NOT EXISTS chunk_b_json JSONB NULL;",
    "ALTER TABLE IF EXISTS dms.document_entities ADD COLUMN IF NOT EXISTS char_start INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_entities ADD COLUMN IF NOT EXISTS char_end INTEGER NULL;",
    "ALTER TABLE IF EXISTS dms.document_entities ADD COLUMN IF NOT EXISTS text_normalized TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_entities ADD COLUMN IF NOT EXISTS canonical_text TEXT NULL;",
    "ALTER TABLE IF EXISTS dms.document_entities ADD COLUMN IF NOT EXISTS source_location_json JSONB NULL;",
    """
    CREATE OR REPLACE VIEW dms.document_document_links AS
    SELECT
        link_id,
        run_id,
        source_document_id,
        target_document_id,
        link_type,
        score,
        raw_link_json AS payload_json,
        created_at,
        updated_at
    FROM dms.document_links;
    """,
    """
    CREATE OR REPLACE VIEW dms.document_entity_links AS
    SELECT
        entity_id AS link_id,
        run_id,
        document_id AS source_document_id,
        entity_id AS target_entity_id,
        entity_type AS link_type,
        NULL::DOUBLE PRECISION AS score,
        payload_json
    FROM dms.document_entities;
    """,
    """
    CREATE OR REPLACE VIEW dms.document_identifier_links AS
    SELECT
        identifier_id AS link_id,
        run_id,
        document_id AS source_document_id,
        identifier_id AS target_identifier_id,
        identifier_type AS link_type,
        confidence_score AS score,
        payload_json
    FROM dms.document_identifiers;
    """,
    """
    CREATE OR REPLACE VIEW dms.document_topic_links AS
    SELECT
        topic_id AS link_id,
        run_id,
        document_id AS source_document_id,
        topic_id AS target_topic_id,
        topic_scope AS link_type,
        score,
        payload_json
    FROM dms.document_topics;
    """,
]
