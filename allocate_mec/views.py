from django.shortcuts import render

from django.http import JsonResponse, HttpResponse
from rest_framework.views import APIView
import json
import pandas as pd
import numpy as np
import re
import gurobipy as gp
from gurobipy import GRB

opt_province_size = None
opt_province_sunni = None

opt_mec_takhsis_shia = None
opt_mec_takhsis_sunni = None

prov_car_model = None
sunni_mec_model = None
shia_mec_model = None

prov_car_model_stat = -10  # means not started
sunni_mec_model_stat = -10
shia_mec_model_stat = -10

final_excel_output_df = None
excel_output = HttpResponse(
    content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
excel_output['Content-Disposition'] = "attachment; filename=Carevan_List.xlsx"

runtime = 0
objbst = 0
objbnd = 0
solcnt = 0
gap = 0.0

program_status = ""
# Options for program status
# "1) Read Alloc Sol"
# "2) Alloc Sol Data Cleaning"
#  "3) Optimization 1/3 - Running Province-Carevan Model"
# "4) Optimization 1/3 Finished - Province-Carevan Model Solution Cleaning"
#  "5) Optimization 2/3 - Running SUNNI Mecca Takhsis Model"
# "6) Optimization 2/3 Finished - SUNNI Mecca Takhsis Model Solution Cleaning"
#  "7) Optimization 3/3 - Running SHIA Mecca Takhsis Model"
# "8) Optimization 3/3 Finished - SHIA Mecca Takhsis Model Solution Cleaning"


class AllocateMecView(APIView):
    def get(self, request, *args, **kwargs):
        global program_status
        global prov_car_model
        global sunni_mec_model
        global shia_mec_model
        try:
            if request.query_params.get("terminate") == "1":
                if program_status[0] == "3":
                    print("Terminating Opt Model 1/3")
                    prov_car_model.terminate()
                elif program_status[0] == "5":
                    print("Terminating Opt Model 2/3")
                    sunni_mec_model.terminate()
                elif program_status[0] == "7":
                    print("Terminating Opt Model 3/3")
                    shia_mec_model.terminate()
                else:
                    return JsonResponse({"Status": program_status})
        except:
            print("Provide terminate parameter in get request")

        return JsonResponse(buildResponse())

    def post(self, request, *args, **kwargs):
        global prov_car_model_stat
        global sunni_mec_model_stat
        global shia_mec_model_stat

        global final_excel_output_df
        global excel_output

        data = request.data
        # Reading API body variables
        param_names = [
            "items_weight", "route_descr", "station_passengers",
            "station_province_index", "province_names", "station_province_size", "station_province_sunni",
            "med_hotel_name", "med_hotel_price",
            "mec_hotel_name", "mec_hotel_price", "mec_hotel_capacity", "mec_hotel_just_sunni", "mec_hotel_reserved_size",
            "province_preference",
            "downhaul_tot_days", "med_first_stay_duration",
            "sunni_hotel_class_toler", "shia_hotel_class_toler"
        ]
        try:
            main_args = [json.loads(data[x]) for x in param_names]
        except Exception as e:
            return JsonResponse({"Error": "Wrong Data Entry: "+str(e)})
        # End Reading API body variables

        try:
            prov_car_model_stat = -10  # means not started
            sunni_mec_model_stat = -10
            shia_mec_model_stat = -10
            final_excel_output_df = main(*main_args)
        except Exception as err:
            # Infeasibility handling
            return JsonResponse({"Error": str(err)})

        # Writing in-memory excel file to download
        final_excel_output_df.to_excel(excel_output, 'کاروان')

        return JsonResponse(final_excel_output_df.to_json(orient='split', force_ascii=False, index=False), safe=False)


def main(items_weight, route_descr, station_passengers,
         station_province_index, province_names, station_province_size, station_province_sunni,
         med_hotel_name, med_hotel_price,
         mec_hotel_name, mec_hotel_price, mec_hotel_capacity, mec_hotel_just_sunni, mec_hotel_reserved_size,
         province_preference,
         downhaul_tot_days, med_first_stay_duration,
         sunni_hotel_class_toler, shia_hotel_class_toler
         ):
    # The stream is equal to sol.ipynb
    global program_status
    global prov_car_model_stat
    global sunni_mec_model_stat
    global shia_mec_model_stat

    # Reading lists from allocate endpoint
    program_status = "1) Read Alloc Sol"
    # med_first_group_index - Medinah Residence
    with open("./Output/Takhsis/med_first_group_index", "r") as fp:
        med_first_group_index = json.load(fp)
    with open("./Output/Takhsis/med_last_group_index", "r") as fp:
        med_last_group_index = json.load(fp)

    # downhaul_grouping_mfirst_index - Parvaz
    with open("./Output/Takhsis/downhaul_grouping_mfirst_index", "r") as fp:
        downhaul_grouping_mfirst_index = json.load(fp)
    with open("./Output/Takhsis/backhaul_grouping_mlast_index", "r") as fp:
        backhaul_grouping_mlast_index = json.load(fp)
    # End Reading lists from allocate endpoint
    # Reading dataframe - the solution of allocate
    df = pd.read_csv("./Output/Takhsis/allocate1_sol.csv")

    # DF Data Cleaning
    program_status = "2) Alloc Sol Data Cleaning"

    def f(s):
        res = re.findall(r'\((\d+), (\d+), (\d+), (\d+)', s)
        return tuple([int(i) for i in res[0]])

    df['Unnamed: 0'] = df['Unnamed: 0'].apply(f)
    df.rename(columns={'Unnamed: 0': 0, '0': 'sol'}, inplace=True)

    # Cell 4
    multi = pd.MultiIndex.from_tuples(df[0])
    df.set_index(multi, inplace=True)
    df.drop(columns=0, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'level_0': 'd', 'level_1': 'w', 'level_2': 'p', 'level_3': 'i'}, inplace=True)
    df['sol'] = df['sol'].round(0).astype(int)

    # Cell 6
    out = pd.DataFrame(columns=['iDay', 'Day', 'iSize', 'Size', 'iMed-Hotel',
                       'Med-Hotel-Index', 'iMec-Group', 'Mec-Group', 'iStation', 'Station'])

    for d in range(df['d'].iloc[-1]+1):
        for w in range(df[df['d'] == 0]['w'].iloc[-1]+1):
            for i in range(sum(df[(df['d'] == d) & (df['w'] == w) & (df['p'] == 1)]['sol'])):
                out = pd.concat([out, pd.DataFrame([[
                    d,
                    d+1,

                    w,
                    items_weight[w],

                    list(np.cumsum(df[(df['d'] == d) & (df['w'] == w)
                                      & (df['p'] == 0)]['sol']) > i).index(True),
                    (med_first_group_index+med_last_group_index)[
                        d][list(np.cumsum(df[(df['d'] == d) & (df['w'] == w) & (df['p'] == 0)]['sol']) > i).index(True)],

                    list(np.cumsum(df[(df['d'] == d) & (df['w'] == w)
                                      & (df['p'] == 1)]['sol']) > i).index(True),
                    list(np.cumsum(df[(df['d'] == d) & (df['w'] == w)
                                      & (df['p'] == 1)]['sol']) > i).index(True)+1,

                    (downhaul_grouping_mfirst_index+backhaul_grouping_mlast_index)[
                        d][list(np.cumsum(df[(df['d'] == d) & (df['w'] == w) & (df['p'] == 2)]['sol']) > i).index(True)],
                    route_descr[(downhaul_grouping_mfirst_index+backhaul_grouping_mlast_index)[d]
                                [list(np.cumsum(df[(df['d'] == d) & (df['w'] == w) & (df['p'] == 2)]['sol']) > i).index(True)]],
                ]],
                    columns=['iDay', 'Day', 'iSize', 'Size', 'iMed-Hotel', 'Med-Hotel-Index', 'iMec-Group', 'Mec-Group', 'iStation', 'Station'])],
                    axis=0, ignore_index=True)

    # Cell 10
    # Inserting carevan sizes that don't exist in station_items with all values equal to 0
    station_items = out.groupby(['iStation', 'Size'])['iSize'].count().unstack(fill_value=0).astype(int)

    to_insert_nonexist_carevan_size = {el: [0]*len(station_items) for el in pd.Series(items_weight)[~pd.Series(
        items_weight).isin(out.groupby(['iStation', 'Size'])['iSize'].count().unstack(fill_value=0).columns)].values}
    station_items = pd.concat([station_items, pd.DataFrame(to_insert_nonexist_carevan_size)], axis=1)
    station_items.sort_index(axis=1, ascending=False, inplace=True)
    station_items.fillna(0, inplace=True)
    station_items = station_items.values.tolist()
    print("station_items")
    print(station_items)
    # Cell 16
    # Optimization Run 1 / 3 in this module - FAST
    program_status = "3) Optimization 1/3 - Running Province-Carevan Model"
    prov_car_model_stat = -5  # means running
    prov_car_model_stat = create_province_carevan_model(station_province_size, station_province_sunni,
                                                        station_items, station_province_index, station_passengers, items_weight)
    if prov_car_model_stat == 3:
        raise Exception("Province Carevan Model is infeasible. Try changing its parameters.")
    program_status = "4) Optimization 1/3 Finished - Province-Carevan Model Solution Cleaning"

    # START tagging province AND sunni to carevans with known station and medina hotel and starting day

    # Sort carevans by Medina hotels index and mec group. The sooner province will be assigned to sooner medina hotels(more expensive)
    # The iMec groups sort is descending, meaning sooner province will go to longer (1 day) carevans
    # Therefore, first provinces will have more expensive package in terms of medina hotels and longer total durations.
    #    This will impose a more expensive mecca hotel later

    # Cell 18
    df_province_size = pd.DataFrame.from_dict(opt_province_size, orient='index')
    multi = pd.MultiIndex.from_tuples(df_province_size.index)
    df_province_size.set_index(multi, inplace=True)
    df_province_size.reset_index(inplace=True)
    df_province_size.rename(columns={'level_0': 'pr', 'level_1': 'w', 0: 'sol'}, inplace=True)
    df_province_size['sol'] = df_province_size['sol'].round(0).astype(int)

    # Cell 19
    def spreadProvinceID(a):
        res = []
        for i in a[a > 0].index:
            res += a[i]*[i]
        return res

    def spreadSunniVal(a):
        res = []
        for i in a[a > 0].index:
            res += a[i]*[1]
        return res

    # Adding medinah hotel price
    out['Med-Hotel-Price'] = out['Med-Hotel-Index'].apply(lambda x: med_hotel_price[x])
    out['Med-Hotel-Name'] = out['Med-Hotel-Index'].apply(lambda x: med_hotel_name[x])

    # Cell 20
    # Sort carevans by Medina hotels price and mec group. The sooner province will be assigned to more expensive medina hotels
    # The iMec groups sort is descending, meaning sooner province will go to longer (1 day) carevans
    # Therefore, first provinces will have more expensive package in terms of medina hotels and longer total durations.
    #   This will impose a more expensive mecca hotel later

    out['iProvince'] = None

    for s in range(len(station_passengers)):
        for w in range(len(items_weight)):
            if df_province_size[df_province_size['pr'].isin(station_province_index[s])].groupby(
                    ['pr', 'w']).sum().sum()[0] == 0:
                continue
            out.loc[out[(out['iStation'] == s) & (out['iSize'] == w)].sort_values(['Med-Hotel-Price', 'iMec-Group'], ascending=[False, False]).index, 'iProvince'] = \
                df_province_size[df_province_size['pr'].isin(station_province_index[s])].groupby(
                    ['pr', 'w']).sum().unstack().apply(spreadProvinceID)[w]

    # Cell 21
    out['Province'] = out['iProvince'].apply(lambda x: province_names[x])

    # Cell 22
    df_province_sunni = pd.DataFrame.from_dict(opt_province_sunni, orient='index')
    multi = pd.MultiIndex.from_tuples(df_province_sunni.index)
    df_province_sunni.set_index(multi, inplace=True)
    df_province_sunni.reset_index(inplace=True)
    df_province_sunni.rename(columns={'level_0': 'pr', 'level_1': 'w', 0: 'sol'}, inplace=True)
    df_province_sunni['sol'] = df_province_sunni['sol'].round(0).astype(int)

    # Cell 23
    # Sort carevans by Medina hotels index and mec group. The sooner carevan will be assigned to later medina hotels(less expensive)
    # The iMec groups sort is descending, meaning sooner province will go to longer (1 day) carevans
    out['Sunni'] = 0

    for s in range(len(station_passengers)):
        for w in range(len(items_weight)):
            if df_province_sunni[df_province_sunni['pr'].isin(station_province_index[s])].groupby(['pr', 'w']).sum()['sol'].sum() > 0:
                sunni_value = df_province_sunni[df_province_sunni['pr'].isin(station_province_index[s])].groupby(
                    ['pr', 'w']).sum().unstack().apply(spreadSunniVal)[w]

                out.loc[
                    out[(out['iStation'] == s) & (out['iSize'] == w)].sort_values(
                        ['iMed-Hotel', 'iMec-Group'], ascending=[True, False]).index[:len(sunni_value)],
                    'Sunni'] = sunni_value

    out['Sunni'] = out['Sunni'].astype(int)
    # END tagging province AND sunni to carevans with known station and medina hotel and starting day

    # Now, building mecca hotel takhsis. Result: Which carevan goes to which hotel in mecca

    # Creating new column: Med-First - MF_Entry/ML_Exit_Day
    mf_down_days = downhaul_tot_days - med_first_stay_duration + 1  # 26
    # Med-First
    out['Med-First'] = 0
    out.loc[out[out['Day'] <= mf_down_days].index, 'Med-First'] = 1
    # MF_Entry/ML_Exit_Day
    out.loc[out[out['Day'] <= mf_down_days].index, 'Day-Code'] = out[out['Day'] <= mf_down_days]['Day']
    out.loc[out[out['Day'] > mf_down_days].index,
            'Day-Code'] = out[out['Day'] > mf_down_days]['Day'] - mf_down_days

    # Cell 109
    # Making sunni index hotel and carevan
    sunni_hotel_index = pd.Series(mec_hotel_just_sunni)
    sunni_hotel_index = sunni_hotel_index[sunni_hotel_index == 1].index.tolist()

    sunni_carevan_index = out[out['Sunni'] == 1].index.tolist()

    # Making shia index hotel and carevan
    shia_hotel_index = pd.Series(mec_hotel_just_sunni)
    shia_hotel_index = shia_hotel_index[shia_hotel_index == 0].index.tolist()

    shia_carevan_index = out[out['Sunni'] == 0].index.tolist()

    # Cell 82
    # In this file in opposite of sol.ipynb, first run sunni, then shia which is longer

    # Run Sunni
    program_status = "5) Optimization 2/3 - Running SUNNI Mecca Takhsis Model"
    sunni_mec_model_stat = -5
    sunni_mec_model_stat = create_mecca_takhsis_model(med_hotel_price, mec_hotel_price, mec_hotel_capacity, mec_hotel_just_sunni, mec_hotel_reserved_size,
                                                      province_preference, sunni_hotel_index, sunni_carevan_index, 1, sunni_hotel_class_toler, out)
    if sunni_mec_model_stat == 3:
        raise Exception("Sunni Mecca Hotel Model is infeasible. Try changing its parameters.")
    program_status = "6) Optimization 2/3 Finished - SUNNI Mecca Takhsis Model Solution Cleaning"
    out['Mec-Hotel-Index'] = None
    # Cell 83
    df_mec_takhsis_sunni = pd.DataFrame.from_dict(opt_mec_takhsis_sunni, orient='index')
    multi = pd.MultiIndex.from_tuples(df_mec_takhsis_sunni.index)
    df_mec_takhsis_sunni.set_index(multi, inplace=True)
    df_mec_takhsis_sunni.reset_index(inplace=True)
    df_mec_takhsis_sunni.rename(columns={'level_0': 'h', 'level_1': 'car', 0: 'sol'}, inplace=True)
    df_mec_takhsis_sunni['sol'] = df_mec_takhsis_sunni['sol'].round(0).astype(int)

    # Cell 84
    res = df_mec_takhsis_sunni[df_mec_takhsis_sunni['sol'] == 1]
    out.loc[[sunni_carevan_index[c] for c in res['car'].values],
            'Mec-Hotel-Index'] = [sunni_hotel_index[h] for h in res['h'].values]

    # Cell 111
    # Run Shia
    program_status = "7) Optimization 3/3 - Running SHIA Mecca Takhsis Model"
    shia_mec_model_stat = -5
    shia_mec_model_stat = create_mecca_takhsis_model(med_hotel_price, mec_hotel_price, mec_hotel_capacity, mec_hotel_just_sunni, mec_hotel_reserved_size,
                                                     province_preference, shia_hotel_index, shia_carevan_index, 0, shia_hotel_class_toler, out)
    if shia_mec_model_stat == 3:
        raise Exception("Shia Mecca Hotel Model is infeasible. Try changing its parameters.")
    program_status = "8) Optimization 3/3 Finished - SHIA Mecca Takhsis Model Solution Cleaning"
    # Cell 90
    df_mec_takhsis_shia = pd.DataFrame.from_dict(opt_mec_takhsis_shia, orient='index')
    multi = pd.MultiIndex.from_tuples(df_mec_takhsis_shia.index)
    df_mec_takhsis_shia.set_index(multi, inplace=True)
    df_mec_takhsis_shia.reset_index(inplace=True)
    df_mec_takhsis_shia.rename(columns={'level_0': 'h', 'level_1': 'car', 0: 'sol'}, inplace=True)
    df_mec_takhsis_shia['sol'] = df_mec_takhsis_shia['sol'].round(0).astype(int)

    # Cell 94
    res = df_mec_takhsis_shia[df_mec_takhsis_shia['sol'] == 1]
    out.loc[[shia_carevan_index[c] for c in res['car'].values],
            'Mec-Hotel-Index'] = [shia_hotel_index[h] for h in res['h'].values]

    # Cell 85
    # Adding Hotel price columns for mec
    out['Mec-Hotel-Price'] = out['Mec-Hotel-Index'].apply(lambda x: mec_hotel_price[x])
    out['Mec-Hotel-Name'] = out['Mec-Hotel-Index'].apply(lambda x: mec_hotel_name[x])

    # Cell 86
    final_output = out[['Med-First', 'Day-Code', 'Mec-Group', 'Size', 'Station', 'Province', 'Sunni', 'Med-Hotel-Index', 'Med-Hotel-Name', 'Med-Hotel-Price',
                       'Mec-Hotel-Index', 'Mec-Hotel-Name', 'Mec-Hotel-Price']]
    # Writing df excel to local directory on server
    final_output.to_excel('./Output/Takhsis/Final_Carevan_List.xlsx', 'کاروان')

    return final_output


# Cell 15
def create_province_carevan_model(target_province_size, target_province_sunni, station_items, station_province_index, station_passengers, items_weight):
    global prov_car_model

    opt_model = gp.Model(name="Province-Carevan Allocation")
    prov_car_model = opt_model

    num_provinces = station_province_index[-1][-1]+1
    num_weight = len(items_weight)
    num_stations = len(station_passengers)

    x = {(p, w): opt_model.addVar(vtype=GRB.INTEGER, lb=0, name='X(%i_%i)' % (p, w))
         for p in range(num_provinces) for w in range(num_weight)}
    sunn = {(p, w): opt_model.addVar(vtype=GRB.INTEGER, lb=0, name='SUN(%i_%i)' % (p, w))
            for p in range(num_provinces) for w in range(num_weight)}

    # Sunni is subset of x (total carevans for each province)
    opt_model.addConstrs(sunn[p, w] <= x[p, w]
                         for p in range(num_provinces) for w in range(num_weight))

    # REMOVED:All station passengers should be assigned to a province
    # opt_model.addConstrs(x[p,w] * items_weight[w] == station_passengers[s]
    #                          for s in range(num_stations) for p in station_province_index[s] for w in range(num_weight))

    # The items weight for a station must be equal to sum of all items weights for its relevant province
    opt_model.addConstrs(gp.quicksum(x[p, w] for p in station_province_index[s]) == station_items[s][w]
                         for s in range(num_stations) for w in range(num_weight))

    # Objective function include 2 statement: 1)Size Error 2)Sunni Error
    # Auxiliary variable for absolute error
    size_error_obj = {(s, p): opt_model.addVar(lb=-1000000, name='SIZE_ERR(%i_%i)' % (s, p))
                      for s in range(num_stations) for p in station_province_index[s]}
    sunni_error_obj = {(s, p): opt_model.addVar(lb=-1000000, name='SUNN_ERR(%i_%i)' % (s, p))
                       for s in range(num_stations) for p in station_province_index[s]}

    opt_model.addConstrs(size_error_obj[s, p] ==
                         target_province_size[s][station_province_index[s].index(p)]
                         - gp.quicksum(x[p, w]*items_weight[w] for w in range(num_weight))
                         for s in range(num_stations) for p in station_province_index[s])

    opt_model.addConstrs(sunni_error_obj[s, p] ==
                         target_province_sunni[s][station_province_index[s].index(p)]
                         - gp.quicksum(sunn[p, w]*items_weight[w] for w in range(num_weight))
                         for s in range(num_stations) for p in station_province_index[s])

    size_abs_error_obj = {(s, p): opt_model.addVar(name='SIZE_ABS_ERR(%i_%i)' % (s, p))
                          for s in range(num_stations) for p in station_province_index[s]}
    sunni_abs_error_obj = {(s, p): opt_model.addVar(name='SUNN_ABS_ERR(%i_%i)' % (s, p))
                           for s in range(num_stations) for p in station_province_index[s]}

    opt_model.addConstrs(size_abs_error_obj[s, p] == gp.abs_(size_error_obj[s, p])
                         for s in range(num_stations) for p in station_province_index[s])
    opt_model.addConstrs(sunni_abs_error_obj[s, p] == gp.abs_(sunni_error_obj[s, p])
                         for s in range(num_stations) for p in station_province_index[s])

    obj1 = gp.quicksum(size_abs_error_obj[s, p]
                       for s in range(num_stations) for p in station_province_index[s])
    obj2 = gp.quicksum(sunni_abs_error_obj[s, p]
                       for s in range(num_stations) for p in station_province_index[s])
    opt_model.ModelSense = GRB.MINIMIZE
    opt_model.setObjective(obj1 + obj2)

    # opt_model.write("out-param.lp")
    # opt_model.write("out-param.mps")

    # Model Parameters
    opt_model.Params.timelimit = 10000
    opt_model.Params.integralityfocus = 1
    # opt_model.Params.mipfocus = 1

    def getSol(model, where):
        if where == GRB.Callback.MIPSOL:
            global opt_province_size
            global opt_province_sunni

            opt_province_size = model.cbGetSolution(x)
            opt_province_sunni = model.cbGetSolution(sunn)
        elif where == GRB.Callback.MIP:
            global runtime
            global objbst
            global objbnd
            global gap

            runtime = model.cbGet(GRB.Callback.RUNTIME)
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            gap = abs(objbst-objbnd)/objbst

    opt_model._vars = opt_model.getVars()
    opt_model.optimize(getSol)

    return opt_model.status

# Cell 110


def create_mecca_takhsis_model(med_hotel_price, mec_hotel_price, mec_hotel_capacity, mec_hotel_just_sunni, mec_hotel_reserved_size,
                               province_preference, sect_hotel_index, sect_carevan_index, sunni, hotel_class_toler, car_df):
    global sunni_mec_model
    global shia_mec_model

    if sunni == 1:
        opt_model = gp.Model(name="Sunni Mecca Hotel Allocation")
        sunni_mec_model = opt_model
    elif sunni == 0:
        opt_model = gp.Model(name="Shia Mecca Hotel Allocation")
        shia_mec_model = opt_model
    else:
        return "Error: Wrong sunni param value"

    num_hotels = len(sect_hotel_index)  # len(mec_hotel_price)
    num_carevans = len(sect_carevan_index)  # len(car_df)

    x = {(h, c): opt_model.addVar(vtype=GRB.BINARY, name='X(%i_%i)' % (h, c))
         for h in range(num_hotels) for c in range(num_carevans)}

    # y means if hotel i is empty or has passengers (semi-full) - just required for objective
    y = {(h): opt_model.addVar(vtype=GRB.BINARY, name='Y(%i)' % (h))
         for h in range(num_hotels)}
    opt_model.addConstrs(y[h] == gp.any_(x[h, c] for c in range(num_carevans))
                         for h in range(num_hotels))

    # Constraints
    # Each carevan should be assigned only to 1 hotel
    opt_model.addConstrs(gp.quicksum(x[h, c] for h in range(num_hotels)) == 1
                         for c in range(num_carevans))

    # Hotel Capacity Constraint
    opt_model.addConstrs(gp.quicksum(x[h, c]*car_df['Size'][sect_carevan_index[c]] for c in range(num_carevans))
                         + gp.quicksum(x[h, c] for c in range(num_carevans)) * mec_hotel_reserved_size[h]
                         <= mec_hotel_capacity[sect_hotel_index[h]]
                         for h in range(num_hotels))

    # Mec price should be >= of med price
    # Find med-first carevans :
    med_last_start = car_df[car_df['Sunni'] == sunni]['Med-First'].reset_index().idxmin()[1]

    # For med first
    car_med_price = car_df[car_df['Sunni'] == sunni]['Med-Hotel-Price'].tolist()
    # ABOVE IS EASIER: car_med_price = car_df['Med-Hotel-Index'].apply(lambda x: med_hotel_price[x]).iloc[sect_carevan_index].tolist()
    opt_model.addConstrs(gp.quicksum(x[h, c] * mec_hotel_price[sect_hotel_index[h]] for h in range(num_hotels)) >= car_med_price[c] - hotel_class_toler[0]
                         for c in range(med_last_start))
    # For med last
    opt_model.addConstrs(gp.quicksum(x[h, c] * mec_hotel_price[sect_hotel_index[h]] for h in range(num_hotels)) >= car_med_price[c] - hotel_class_toler[1]
                         for c in range(med_last_start, num_carevans))

    # Limit just-sunni hotels
    # just_sunni_hotels= pd.Series(mec_hotel_just_sunni)
    # just_sunni_hotels = just_sunni_hotels[just_sunni_hotels==1].index.tolist()

    # non_sunni_carevans = car_df[car_df['Sunni'] == 0].index.tolist()

    # opt_model.addConstrs(gp.quicksum(x[h,c] for c in non_sunni_carevans) == 0
    #                     for h in just_sunni_hotels)

    # Setting objective function
    # obj1 is for cost reduction - making some hotels (more expensive) empty as much as possible
    obj1 = gp.quicksum(y[h]*mec_hotel_price[sect_hotel_index[h]] for h in range(num_hotels))

    # obj2 is for preference of some provinces be together
    # z[h,pref] means hotel h contains preferenc pref if == 1, otherwise 0
    num_prov_pref = len(province_preference)

    z = {(h, pref): opt_model.addVar(vtype=GRB.BINARY, name='Z(%i_%i)' % (h, pref))
         for h in range(num_hotels) for pref in range(num_prov_pref)}

    opt_model.addConstrs(z[h, pref] == gp.any_(x[h, c]
                                               for c in [sect_carevan_index.index(i) for i in
                                                         car_df[(car_df['iProvince'].isin(province_preference[pref])) & (car_df['Sunni'] == sunni)].index]
                                               )
                         for h in range(num_hotels) for pref in range(num_prov_pref))

    # zhotel equals number of preference existing in hotel h - 1 . It should be close to only 1
    zhotel = {(h): opt_model.addVar(vtype=GRB.INTEGER, name='ZH(%i)' % (h))
              for h in range(num_hotels)}

    opt_model.addConstrs(zhotel[h] == gp.quicksum(z[h, pref] for pref in range(num_prov_pref)) - 1
                         for h in range(num_hotels))

    # zhotel_abs is the absolute value of |zhotel - 1|
    zhotel_abs = {(h): opt_model.addVar(vtype=GRB.INTEGER, name='ZH_ABS(%i)' % (h))
                  for h in range(num_hotels)}

    opt_model.addConstrs(zhotel_abs[h] == gp.abs_(zhotel[h])
                         for h in range(num_hotels))

    obj2 = gp.quicksum(zhotel_abs[h] for h in range(num_hotels))

    # Setting obj and confuguration
    opt_model.ModelSense = GRB.MINIMIZE
    opt_model.setObjective(3*obj1 + obj2)

    # opt_model.write("out-param.lp")
    # opt_model.write("mec_alloc_out_param.mps")

    # Model Parameters
    opt_model.Params.timelimit = 25000
    opt_model.Params.integralityfocus = 1
    opt_model.Params.mipfocus = 1
    # opt_model.Params.heuristics = 0.2

    def getSol(model, where):
        global opt_mec_takhsis_sunni
        global opt_mec_takhsis_shia

        if where == GRB.Callback.MIPSOL:
            if sunni == 1:
                opt_mec_takhsis_sunni = model.cbGetSolution(x)
            else:
                opt_mec_takhsis_shia = model.cbGetSolution(x)
        elif where == GRB.Callback.MIP:
            global runtime
            global objbst
            global objbnd
            global gap

            runtime = model.cbGet(GRB.Callback.RUNTIME)
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            gap = (objbst-objbnd)/objbst

    opt_model._vars = opt_model.getVars()
    opt_model.optimize(getSol)

    return opt_model.status


def buildResponse():
    global program_status
    global prov_car_model_stat
    global sunni_mec_model_stat
    global shia_mec_model_stat

    global runtime
    global objbst
    global objbnd
    global solcnt
    global gap

    def optStatus(s):
        out = ""
        if s == -10:
            out = "Not Started"
        elif s == 2:
            out = "Optimal"
        elif s == 3:
            out = "Infeasible"
        elif s == 11:
            out = "Interrupted"
        else:
            out = "Running"
        return out

    resp = {"Program Status": program_status,
            "Province Carevan Model Status": optStatus(prov_car_model_stat),
            "Sunni Mecca Hotel Model Status": optStatus(sunni_mec_model_stat),
            "Shia Mecca Hotel Model Status": optStatus(shia_mec_model_stat),
            "runtime": runtime, "objbst": objbst, "objbnd": objbnd, "solcnt": solcnt,
            "gap": gap, }
    # stat_num = int(program_status[0])
    # try:
    #     if stat_num >= 7:
    #         resp["Shia Mecca Hotel Model Status"] = optStatus(shia_mec_model_stat)
    #     if stat_num >= 5:
    #         resp["Sunni Mecca Hotel Model Status"] = optStatus(sunni_mec_model.status)
    #     if stat_num >= 3:
    #         resp["Province Carevan Model Status"] = optStatus(prov_car_model.status)
    # except:
    #     print("Error: Opt Model has no status")

    return resp


class AllocMecExcelView(APIView):
    def get(self, request, *args, **kwargs):
        global excel_output
        global shia_mec_model

        # try:
        #     if shia_mec_model.solcnt == 0:
        #         return HttpResponse("No Results Yet")
        # except:
        #     return HttpResponse("Final optimization module has not started yet.")

        return excel_output
