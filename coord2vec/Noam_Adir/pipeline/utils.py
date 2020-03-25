import pandas as pd


def get_non_repeating_coords(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    returns a new DataFrame only with non repeating coordinants
    will take the first row a specific coordinate appeared in for all coordinates
    Args:
        full_df: if column "coord_id" exists uses this col to choose rows
                if not assume column "coord" exxist and creates a column "coord_id"

    Returns: DataFrame with column coord_id with unique id for each row

    """
    df = full_df.copy()
    if "coord_id" not in df.columns:
        coord_2coordid = {coord: i for i, coord in enumerate(df["coord"].unique())}
        df["coord_id"] = df.apply(lambda row: coord_2coordid[row["coord"]], axis=1)
    apeared_coord_id = {}
    coord_ids = df["coord_id"]
    for inx in df.index:
        if coord_ids[inx] in apeared_coord_id:
            df = df.drop([inx])
        else:
            apeared_coord_id[coord_ids[inx]] = True
    #     print("test sould be one:", len(set(apeared_coord_id.values())))
    return df
