import pandas as pd


# remove the hierachy column produced by pivot_table
def recolumns(data):

    columns=[]
    column=data.columns

    for cl in column:
        columns.append(cl[1])

    data.columns=columns

    return data


# get the original data and process them
def original_data(sale_path, lead_path, stock_path):

    sale = pd.read_csv(sale_path)
    leadtime = pd.read_csv(lead_path)
    initial_stock = pd.read_csv(stock_path)

    sale["date"]=pd.to_datetime(sale["date"],format="%Y/%m/%d")
    sale_pivoted = pd.pivot_table(sale,values=["sale"],index=["date"],columns=["idx"])
    sale_pivoted = recolumns(sale_pivoted).fillna(0)
    leadtime["date"]=pd.to_datetime(leadtime["date"],format="%Y/%m/%d")
    lead_pivoted = pd.pivot_table(leadtime,values=["leadtime"],index=["date"],columns=["idx"])
    lead_pivoted = recolumns(lead_pivoted).fillna(0)

    initial_stock['index'] = initial_stock['dc_id'] + '_' + initial_stock['sku_id']
    initial_stock.set_index('index', inplace=True)
    initial_stock = initial_stock[['available_inv']].fillna(0)
    
    date = lead_pivoted.index
    # remove the colunms with all zero values in the date index
    sale_pivoted_date = sale_pivoted.loc[date]
    valid_dcsku = sale_pivoted_date.columns[(sale_pivoted_date != 0).any(axis=0)]
    sale_pivoted = sale_pivoted.loc[:, valid_dcsku]
    lead_pivoted = lead_pivoted.loc[:, valid_dcsku]
    initial_stock = initial_stock.loc[valid_dcsku]

    return sale_pivoted, lead_pivoted, initial_stock


if __name__ == "__main__":
    sale_pivoted, lead_pivoted, initial_stock = original_data("data/original_real_data/sales_data.csv", "data/original_real_data/leadtime_data.csv", "data/original_real_data/dc_inventory.csv")
    sale_pivoted.to_csv("data/real_data/sales.csv", index=True)
    lead_pivoted.to_csv("data/real_data/lead.csv", index=True)
    initial_stock.to_csv("data/real_data/instock.csv", index=True)