def to_lowercase(
    pipeline, colname_to_replace, output_table_name, input_table_name="df"
):
    sql = f"""
    select
        * exclude ({colname_to_replace}),
        list_transform({colname_to_replace}, x -> lower(x))
            as {colname_to_replace}
    from {input_table_name}
    """

    pipeline.enqueue_sql(sql, output_table_name)
    return pipeline