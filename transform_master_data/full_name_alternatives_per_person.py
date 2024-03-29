def add_full_name_alternatives_per_person(
    pipeline, output_table_name, input_table_name="df"
):

    # Ensure humanAltLabel is a (potentially empty) array
    sql = f"""
    select
        *,
        case when
            array_length(humanAltLabel) = 0 then []
            else str_split(humanAltLabel[1], ', ')
        end as humanAltLabel_fix,
        case when
            array_length(pseudonym) = 0 then []
            else str_split(pseudonym[1], ', ')
        end as pseudonym_fix,

    from {input_table_name}
    """

    pipeline.enqueue_sql(sql, "rel_human_alt_label_array_fixed")

    # Create a list of all the human labels and human alt labels
    sql = """
    select
        * EXCLUDE (humanAltLabel_fix, pseudonym_fix),
        list_concat(list_concat(humanLabel,  humanAltLabel_fix), pseudonym_fix) as full_name_arr
    from rel_human_alt_label_array_fixed
    """

    pipeline.enqueue_sql(sql, "rel_human_labels_as_array")

    # Remove the occasional entry which is actually the persons ID e.g. Q85910376
    sql = """
    select
        * EXCLUDE (full_name_arr),
        array_filter(full_name_arr, x -> (not regexp_matches(x, '^Q\d+$')))
            as full_name_arr,
    from rel_human_labels_as_array
    """

    pipeline.enqueue_sql(sql, "rel_human_labels_as_array_filtered_qnumbers")

    # Use given name and family name to crate a name string e.g. given_name [John,James]
    # and family name [Smith]
    # become a single string "John James Smith"
    sql = """
    select
        *,
        replace(list_string_agg(list_concat(given_nameLabel, family_nameLabel)),
                ',', ' ')
            as family_give_name_concat,

    from rel_human_labels_as_array_filtered_qnumbers
    """

    pipeline.enqueue_sql(sql, "concatenated_given_family_names")

    # Add this to our list of full names
    sql = """
    select
        * EXCLUDE (full_name_arr, family_give_name_concat),
        list_concat(full_name_arr,  list_value(family_give_name_concat))
            as full_name_arr,
    from concatenated_given_family_names
    """

    pipeline.enqueue_sql(sql, "all_names_in_list")

    # Deduplicate the list of full names
    sql = """
    select
        * EXCLUDE (full_name_arr),
        list_distinct(full_name_arr) as full_name_arr,
    from all_names_in_list
    """

    pipeline.enqueue_sql(sql, output_table_name)

    return pipeline
