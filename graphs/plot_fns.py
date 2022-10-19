import numpy as np
import altair as alt
import pandas as pd
import math

def get_discrete_data(df, var_name, cluster_size=None):
    df_all = df.copy()
    if cluster_size:
        f1 = df_all["vertices"] == cluster_size
        df_all = df_all[f1].copy()

    abs_ = df_all.groupby([var_name, "any_false_matches"]).size()
    percs = (abs_ / abs_.groupby(level=0).sum()).reset_index(name="perc")
    abs_ = abs_.reset_index(name="count")
    results = abs_.merge(
        percs, on=[f"{var_name}", "any_false_matches"]
    )

    return results


def get_discrete_chart(df, var_name, filter_out_good_clusters=False):

    ax = alt.Axis(title=f"{var_name}")

    chart_base = (
        alt.Chart(df)
        .transform_stack(
            stack="perc",
            as_=["perc_1", "perc_2"],
            groupby=[f"{var_name}"],
            offset="normalize",
            sort=[
                alt.SortField(
                    "any_false_matches", order="descending"
                )
            ],
        )
        .encode()
    )

    if filter_out_good_clusters:
        c1 = chart_base.mark_bar().transform_filter(
            "datum.any_false_matches==true"
        )
    else:
        c1 = chart_base.mark_bar()

    c1 = c1.encode(
        y=alt.Y(
            "perc_1:Q",
            axis=alt.Axis(title="% Clusters with false positives", format=",.1%"),
        ),
        y2=alt.Y2("perc_2:Q"),
        x=alt.X(f"{var_name}:O", axis=ax),
        tooltip=[
            alt.Tooltip("perc:Q", format=",.2%"),
            alt.Tooltip("count:Q", format=",.0f"),
            "any_false_matches",
        ],
        color="any_false_matches",
    ).properties(title=f"Clusters containing FP by {var_name}")

    c2 = (
        chart_base.mark_bar()
        .transform_filter("datum.any_false_matches==true")
        .encode(
            y=alt.Y(
                "count:Q", axis=alt.Axis(title="Num clusters with FP", format=",.0f")
            ),
            x=alt.X(f"{var_name}:O", axis=ax),
            tooltip=[
                alt.Tooltip("perc:Q", format=",.2%"),
                alt.Tooltip("count:Q", format=",.0f"),
                "any_false_matches",
            ],
            color="any_false_matches",
        )
        .properties(height=100)
    )

    c3 = (
        chart_base.mark_bar()
        .transform_filter("datum.any_false_matches==false")
        .encode(
            y=alt.Y(
                "count:Q", axis=alt.Axis(title="Num clusters with no FP", format=",.0f")
            ),
            x=alt.X(f"{var_name}:O", axis=ax),
            tooltip=[
                alt.Tooltip("perc:Q", format=",.2%"),
                alt.Tooltip("count:Q", format=",.0f"),
                "any_false_matches",
            ],
            color="any_false_matches",
        )
        .properties(height=100)
    )

    return c1 & c2 & c3



def get_binned_data(
    df, var_name, step=0.1, cluster_size=None, cut_fn="cut", num_bins=10
):

    df_all = df.copy()
    if cluster_size:
        f1 = df_all["vertices"] == cluster_size
        df_all = df_all[f1].copy()

    min_ = df_all[var_name].min()
    max_ = df_all[var_name].max()
    min_ = math.floor(min_ * 10 - 0.00001) / 10
    max_ = math.ceil(max_ * 10) / 10
    if cut_fn == "cut":

        bins = np.arange(min_, max_ + step, step)
        df_all[f"{var_name}_binned"] = pd.cut(df_all[var_name], bins=bins)

    if cut_fn == "qcut":
        df_all[f"{var_name}_binned"] = pd.qcut(
            df_all[var_name], num_bins, duplicates="drop"
        )

    gvars = [f"{var_name}_binned", "any_false_matches"]

    abs_ = df_all[gvars].groupby(gvars).size()

    percs = (abs_ / abs_.groupby(level=0).sum()).reset_index(name="perc")
    abs_ = abs_.reset_index(name="count")

    results = abs_.merge(
        percs, on=[f"{var_name}_binned", "any_false_matches"]
    )
    results[f"{var_name}_binned_low"] = results[f"{var_name}_binned"].apply(
        lambda x: x.left
    )
    results[f"{var_name}_binned_high"] = results[f"{var_name}_binned"].apply(
        lambda x: x.right - ((max_ - min_) / 200)
    )

    results = results.drop(f"{var_name}_binned", axis=1)
    return results


def get_binned_chart(
    df, var_name, filter_out_good_clusters=False, average_fp_value=None
):

    if average_fp_value:
        df["average_fp_value"] = average_fp_value
    ax = alt.Axis(title=f"Binned {var_name}")

    min_ = df[f"{var_name}_binned_low"].min()
    max_ = df[f"{var_name}_binned_high"].max()
    min_ = math.floor(min_ * 10) / 10
    max_ = math.ceil(max_ * 10) / 10

    sc = alt.Scale(padding=0.1, nice=False, domain=[min_, max_])

    chart_base = (
        alt.Chart(df)
        .transform_stack(
            stack="perc",
            as_=["perc_1", "perc_2"],
            groupby=[f"{var_name}_binned_low", f"{var_name}_binned_high"],
            offset="normalize",
            sort=[
                alt.SortField(
                    "any_false_matches", order="descending"
                )
            ],
        )
        .encode()
    )

    if filter_out_good_clusters:
        c1 = chart_base.mark_bar().transform_filter(
            "datum.any_false_matches==true"
        )
    else:
        c1 = chart_base.mark_bar()

    c1 = c1.encode(
        y=alt.Y(
            "perc_1:Q",
            axis=alt.Axis(title="% Clusters with false positives", format=",.1%"),
        ),
        y2=alt.Y2("perc_2:Q"),
        x=alt.X(f"{var_name}_binned_low:Q", scale=sc, axis=ax),
        x2=alt.X2(f"{var_name}_binned_high:Q"),
        tooltip=[
            alt.Tooltip("perc:Q", format=",.2%"),
            alt.Tooltip("count:Q", format=",.0f"),
            "any_false_matches",
        ],
        color="any_false_matches",
    ).properties(
        title=f"Distribution of clusters containing false positives by {var_name}"
    )
    if average_fp_value:
        c1_line = chart_base.mark_rule(strokeDash=[4, 4]).encode(
            y="average_fp_value", color=alt.value("red"), size=alt.value(2)
        )

        c1_text = chart_base.mark_text(yOffset=-10, xOffset=250).encode(
            y="average(average_fp_value)",
            text=alt.value(f"Average FP within cluster size"),
        )
        c1 = c1 + c1_line + c1_text

    c2 = (
        chart_base.mark_bar()
        .transform_filter("datum.any_false_matches==true")
        .encode(
            y=alt.Y(
                "count:Q", axis=alt.Axis(title="Num clusters with FP", format=",.0f")
            ),
            x=alt.X(f"{var_name}_binned_low:Q", scale=sc, axis=ax),
            x2=alt.X2(f"{var_name}_binned_high:Q"),
            tooltip=[
                alt.Tooltip("perc:Q", format=",.2%"),
                alt.Tooltip("count:Q", format=",.0f"),
                "any_false_matches",
            ],
            color="any_false_matches",
        )
        .properties(height=100)
    )

    c3 = (
        chart_base.mark_bar()
        .transform_filter("datum.any_false_matches==false")
        .encode(
            y=alt.Y(
                "count:Q", axis=alt.Axis(title="Num clusters with no FP", format=",.0f")
            ),
            x=alt.X(f"{var_name}_binned_low:Q", scale=sc, axis=ax),
            x2=alt.X2(f"{var_name}_binned_high:Q"),
            tooltip=[
                alt.Tooltip("perc:Q", format=",.2%"),
                alt.Tooltip("count:Q", format=",.0f"),
                "any_false_matches",
            ],
            color="any_false_matches",
        )
        .properties(height=100)
    )

    return c1 & c2 & c3