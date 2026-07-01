import pandas as pd

REFERENCE_DATE = pd.Timestamp("2026-06-28")

#Read salesforce data
def readSalesForceData():
    df = pd.read_csv("salesforce_data.csv")
    df['donation_date'] = pd.to_datetime(df["donation_date"])
    print("Retrieved salesforce data and formatted date column")
    return df

#Read netsuite data
def readNetSuiteData():
    df = pd.read_csv("netsuite_data.csv")
    df['payment_date'] = pd.to_datetime(df["payment_date"])
    print("Retrieved Netsuite data and formatted date column")
    return df

#Read marketing engagement data
def readMarketingData():
    df = pd.read_csv("marketing_data.csv")
    df['email_sent_date'] = pd.to_datetime(df["email_sent_date"])
    print("Retrieved marketing data and formatted date column")
    return df

def validateSFData(df,table):
    print(f"Doing data checks for {table} table")
    df = df.dropna(subset = ['donor_id','donation_amount'])
    df = df[df["donation_amount"]>=0]
    df = df.drop_duplicates()
    df["channel"] = df["channel"].str.strip()
    df["campaign_name"] = df["campaign_name"].str.strip()
    df["donor_id"] = df["donor_id"].astype(int)
    return df

def validateNetSuiteData(df,table):
    print(f"Doing data checks for {table} table")
    df = df.dropna(subset = ["donor_id","amount_paid"])
    df= df[df["amount_paid"]>=0]
    df = df.drop_duplicates()
    df["donor_id"] = df["donor_id"].astype(int)
    return df

def validateMarketingData(df,table):
    print(f"Doing data checks for {table} table")
    df = df.dropna(subset=["donor_id"])
    df = df.drop_duplicates()
    df["donor_id"] = df["donor_id"].astype(int)
    return df


def aggregateAllDonorIDs(salesforce_df,netsuite_df,marketing_df):
    donor_ids = pd.concat([
        salesforce_df[["donor_id"]],
        netsuite_df[["donor_id"]],
        marketing_df[["donor_id"]]
    ]).drop_duplicates().reset_index(drop=True)
    donor_ids["reference_date"]= REFERENCE_DATE
    return donor_ids


def buildRFMFeatures(salesforce_df):
    rfm = salesforce_df.groupby("donor_id").agg(
        last_donation_date = ("donation_date","max"),
        donation_count = ("donation_amount","count"),
        total_amount_donated = ("donation_amount","sum"),
        avg_donation = ("donation_amount","mean"),
        max_single_donation = ("donation_amount","max")
    ).reset_index()

    #Get number of days between last donation date of donor and reference date
    rfm["recency_days"] = (REFERENCE_DATE - rfm["last_donation_date"]).dt.days

    #To get list of donors who donated from last one year w.r.t reference date
    date_1yr = REFERENCE_DATE - pd.DateOffset(days=365)
    df_1yr = salesforce_df[salesforce_df["donation_date"] >= date_1yr]

    rfm_1yr = df_1yr.groupby("donor_id").agg(
        donation_count_1yr = ("donation_amount","count"),
        total_amount_donated_1yr = ("donation_amount","sum"),
    ).reset_index()

    #Joining two tables to get list of donors and their one year metrics 
    rfm = rfm.merge(rfm_1yr, on="donor_id", how="left")

    return rfm


def buildBehaviorFeatures(salesforce_df):
    df = salesforce_df.sort_values(["donor_id","donation_date"])

    #Get days gap between two consecutive donations
    df["prev_donation_date"]= df.groupby("donor_id")["donation_date"].shift(1)
    df["days_gap"]= (df["donation_date"] - df["prev_donation_date"]).dt.days

    #add mean of days_gap to behavior df 
    behavior = df.groupby("donor_id").agg(
        avg_days_between_donations = ("days_gap","mean"),
        is_recurring = ("donation_type", lambda x : ((x == "recurring").any()))
    ).reset_index()

    #calculate recency days to determine lapsed flag value(1/0)
    recency = salesforce_df.groupby("donor_id")["donation_date"].max().reset_index()
    recency["recency_days"] = (REFERENCE_DATE - recency["donation_date"]).dt.days

    #merge behavior and recency on left join
    behavior = behavior.merge(recency[["donor_id","recency_days"]],on="donor_id",how="left")

    behavior["lapsed_flag"]= (behavior["recency_days"]>365).astype(int)
    behavior = behavior.drop(columns=["recency_days"])

    #see the trend for last 12 months and prior 12 months for total amount donated
    date_1yr = REFERENCE_DATE - pd.DateOffset(days=365)
    date_2yr = REFERENCE_DATE - pd.DateOffset(days=730)

    df_1yr = salesforce_df[salesforce_df["donation_date"]>=date_1yr].groupby("donor_id")["donation_amount"].sum()
    df_2yr = salesforce_df[(salesforce_df["donation_date"]>=date_2yr) & 
                           (salesforce_df["donation_date"]<date_1yr)].groupby("donor_id")["donation_amount"].sum()
    
    trend = pd.DataFrame({"df_1yr":df_1yr,"df_2yr":df_2yr}).fillna(0)
    #giving trend as up/down/flat based on donation amount for last 2 years
    trend["giving_trend"] = trend.apply(
        lambda x: "up" if x["df_1yr"] > x["df_2yr"]
                  else ("down" if x["df_1yr"] < x["df_2yr"] else "flat"),axis=1
    )

    trend = trend[["giving_trend"]].reset_index()

    behavior = behavior.merge(trend,on="donor_id",how="left")
    behavior["giving_trend"] = behavior["giving_trend"].fillna("flat")

    return behavior

def buildFinancialFeatures(salesforce_df,netsuite_df):
    sf_df = salesforce_df[["donor_id","donation_date","donation_amount"]].rename(columns ={"donation_amount":"amount_paid"})
    merged_df = netsuite_df.merge(sf_df,on=["donor_id","amount_paid"],how="left")
    #calculate avg number of days to settle payment for each donor
    merged_df["days_to_settle"] = (merged_df["payment_date"]-merged_df["donation_date"]).dt.days
    avg_settle = (merged_df.groupby("donor_id")["days_to_settle"]
                           .mean()
                           .reset_index()
                           .rename(columns={"days_to_settle":"avg_days_to_settle"}))
    #get total payments and total refunds to cal refund rate
    financial = netsuite_df.groupby("donor_id").agg(
        total_payments = ("amount_paid","count"),
        total_refunds = ("refund_flag","sum"),
    ).reset_index()
    financial["refund_rate"]= financial["total_refunds"] / financial["total_payments"]

    prefPayment_df = (netsuite_df.groupby("donor_id")["payment_method"]
                      .agg(lambda x: x.mode()[0])
                      .reset_index()
                      .rename(columns={"payment_method":"preferred_payment_method"}))
    
    financial = financial.merge(prefPayment_df,on="donor_id",how="left")
    financial = financial.merge(avg_settle,on="donor_id",how="left")

    return financial


def buildMarketingFeatures(marketing_df):
    engagement = marketing_df.groupby("donor_id").agg(
                 emails_sent = ("email_opened","count"),
                 total_emails_opened = ("email_opened","sum"),
                 total_clicks = ("link_clicked","sum"),
                 events_attended = ("event_attended","sum"),
                 is_unsubscribed = ("unsubscribed","max"),
    ).reset_index()

    #calculate emails open rate and click rate
    engagement["open_rate"] = engagement["total_emails_opened"] / engagement["emails_sent"]
    engagement["click_rate"] = engagement["total_clicks"] / engagement["emails_sent"]

    #drop columns total_emails_opened,total_clicks as these are not necessary
    engagement = engagement.drop(columns=["total_emails_opened","total_clicks"])

    #get last email open date for donors where email_opened == 1
    last_open_df = (marketing_df[marketing_df["email_opened"]==1]
                    .groupby("donor_id")["email_sent_date"]
                    .max()
                    .reset_index()
                    .rename(columns={"email_sent_date":"last_open_date"}))
    
    last_open_df["days_since_last_open"] = (REFERENCE_DATE - last_open_df["last_open_date"]).dt.days
    engagement = engagement.merge(last_open_df[["donor_id","days_since_last_open"]],on="donor_id",how="left")

    return engagement


def buildCampaignFeatures(salesforce_df):
    campaign = salesforce_df.groupby("donor_id").agg(
        unique_campaigns_count = ("campaign_name","nunique"),
        unique_channels_count = ("channel","nunique"),
    ).reset_index()

    campaign["multi_channel_flag"] = (campaign["unique_channels_count"]>1).astype(int)

    #get first and last channel for each donor by sorting with donation date 
    sort_df = salesforce_df.sort_values("donation_date")
    first_channel_df = sort_df.groupby("donor_id")["channel"].first().reset_index().rename(columns={"channel":"first_channel"})
    last_channel_df = sort_df.groupby("donor_id")["channel"].last().reset_index().rename(columns={"channel":"last_channel"})

    campaign = campaign.merge(first_channel_df,on="donor_id",how="left")
    campaign = campaign.merge(last_channel_df,on="donor_id",how="left")
    return campaign


def joinAllFeatures(donorIds,rfm,behavior,financing,marketing,campaign):  
    feature_df = donorIds.copy()
    feature_df = feature_df.merge(rfm,on="donor_id",how="left")
    feature_df = feature_df.merge(behavior,on="donor_id",how="left")
    feature_df = feature_df.merge(financing,on="donor_id",how="left")
    feature_df = feature_df.merge(marketing,on="donor_id",how="left")
    feature_df = feature_df.merge(campaign,on="donor_id",how="left")

    #list of columns to flag with 0 with no records in that source
    flag_col = [
         "donation_count", "donation_count_1yr", "total_amount_donated_1yr",
        "is_recurring",
        "total_payments", "total_refunds",
        "emails_sent", "events_attended", "is_unsubscribed",
        "multi_channel_flag"
    ]
    #fill 0 for each column listed in flag_col in feature_df
    for col in flag_col:
        if col in feature_df:
            feature_df[col] = feature_df[col].fillna(0)

    return feature_df


def validate_feature_df(feature_df,donorIds) :
    issues = 0

    #check count of rows in feature_df matches actual number of donors
    expected_rows = donorIds["donor_id"].nunique()
    actual_rows = len(feature_df)
    if actual_rows != expected_rows:
        print(f"Rows count is not matched, expected : {expected_rows} but actual : {actual_rows}")
        issues +=1
    else:
        print(f"Rows count is matched, count : {actual_rows}")

    #check if nulls exits in critical columns 
    null_count = feature_df.isnull().sum()
    null_percentage = null_count/len(feature_df) * 100
    cri_col = ["donor_id", "recency_days", "total_amount_donated", "donation_count"]
    for col in cri_col:
        if col in feature_df and null_percentage[col] >0:
            print(f"{col} has {null_percentage[col]:.1f}% nulls")
            issues+=1

    #check if recency_days>0 for all rows
    if "recency_days" in feature_df.columns:
        negative_recency = (feature_df["recency_days"]<0).sum()
        if negative_recency >0:
            print(f"recency_days has {negative_recency} negative values")
            issues += 1
    
    #check rate columns value is in between 0 and 1
    rate_col = ["open_rate", "click_rate", "refund_rate"]

    for col in rate_col:
        if col in feature_df.columns:
            not_in_range = feature_df[col].dropna()
            not_in_range = ((not_in_range < 0) | (not_in_range > 1)).sum()
            if not_in_range >0:
                print(f"{col} has {not_in_range} values outside [0,1]") 
                issues +=1

    if issues ==0:
        print("All checks are passed. ")
    else:
        print(f"{issues} issues found - review the data.")


def save_feature_toCSV(feature_df) :
    feature_df.to_csv("feature_table.csv", index=False)
    print("Saved feature store to feature_table.csv")


def featurePipeline():
    #read data from all 3 source files
    salesforce_df = readSalesForceData()
    netsuite_df = readNetSuiteData()
    marketing_df = readMarketingData()

    #validate data 
    salesforce_df = validateSFData(salesforce_df,"Salesforce")
    netsuite_df = validateNetSuiteData(netsuite_df,"Netsuite")
    marketing_df = validateMarketingData(marketing_df,"Marketing")

    ##To get list of all unique donorId's from 3 tables
    donorIds = aggregateAllDonorIDs(salesforce_df,netsuite_df,marketing_df)

    #build features with all data files
    rfm = buildRFMFeatures(salesforce_df)
    behavior = buildBehaviorFeatures(salesforce_df)
    financing = buildFinancialFeatures(salesforce_df,netsuite_df)
    marketing = buildMarketingFeatures(marketing_df)
    campaign = buildCampaignFeatures(salesforce_df)

    #join all feature groups
    feature_df = joinAllFeatures(donorIds,rfm,behavior,financing,marketing,campaign) 

    #validate data in feature_df
    validate_feature_df(feature_df,donorIds)

    #save feature_df to csv
    save_feature_toCSV(feature_df) 

    print("Final feature data:",feature_df.to_string())


if __name__ == "__main__":
   result = featurePipeline()
