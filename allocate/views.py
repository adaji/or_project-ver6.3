# from pyexpat import model
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from rest_framework.views import APIView
import json

# My Optimization libraries
from . import total_view_visualize as model
import numpy as np

# Global var
prob = None
med_first_prob = None
med_last_prob = None
items_weight = None
excel_output = HttpResponse(
    content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
excel_output['Content-Disposition'] = "attachment; filename=Total_Res_View.xlsx"


class AllocateView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            if request.query_params.get('terminate') == "1":
                print('Terminating by Get')
                model.opt_model.terminate()
        except:
            print("Provide terminate parameter in get request")

        return JsonResponse(buildResponse())

    def post(self, request, *args, **kwargs):
        data = request.data
        global prob
        global med_first_prob
        global med_last_prob
        global items_weight
        global excel_output
        # Serialize
        # problem_names = [
        #     "med_first_tot_pass",
        #     "med_first_stay_duration",
        #     "downhaul_tot_days",
        #     "downhaul_actual_daily_flight_med",
        #     "downhaul_actual_daily_flight_mec",
        #     "med_first_hotel_price",
        #     "med_first_hotel_capacity",
        #     "med_first_res_solution",

        #     "med_last_tot_pass",
        #     "med_last_stay_duration",
        #     "backhaul_tot_days",
        #     "backhaul_actual_daily_flight_med",
        #     "backhaul_actual_daily_flight_mec"
        #     "med_last_hotel_price",
        #     "med_last_hotel_capacity",
        #     "med_last_res_solution",

        #     "down_end_to_back_start_days",
        #     "med_last_downhaul_start_day",
        #     "med_last_backhaul_start_day",

        #     "items_weight",
        #     "items_value",
        # ]
        try:
            med_first_prob = model.ResProblem(
                tot_passengers=json.loads(data["med_first_tot_pass"]),
                stay_duration=json.loads(data["med_first_stay_duration"]),
                tot_days=json.loads(data["downhaul_tot_days"]),
                hotel_price=json.loads(data["med_first_hotel_price"]),
                hotel_capacity=json.loads(data["med_first_hotel_capacity"]),
                daily_flight_cap=json.loads(data["downhaul_actual_daily_flight_med"]),
            )

            med_last_prob = model.ResProblem(
                tot_passengers=json.loads(data["med_last_tot_pass"]),
                stay_duration=json.loads(data["med_last_stay_duration"]),
                tot_days=json.loads(data["backhaul_tot_days"]),
                hotel_price=json.loads(data["med_last_hotel_price"]),
                hotel_capacity=json.loads(data["med_last_hotel_capacity"]),
                daily_flight_cap=json.loads(data["backhaul_actual_daily_flight_med"]),
            )

            downhaul_sol = json.loads(data["downhaul_sol"])
            downhaul_sol_res = json.loads(data["downhaul_sol_res"])
            downhaul_med_staff_reserve = json.loads(data["downhaul_med_staff_reserve"])
            downhaul_jed_staff_reserve = json.loads(data["downhaul_jed_staff_reserve"])
            backhaul_sol = json.loads(data["backhaul_sol"])
            backhaul_sol_res = json.loads(data["backhaul_sol_res"])
            backhaul_med_staff_reserve = json.loads(data["backhaul_med_staff_reserve"])
            backhaul_jed_staff_reserve = json.loads(data["backhaul_jed_staff_reserve"])
            flight_capacity = json.loads(data["flight_capacity"])

            downhaul_actual_daily_flight_mec = json.loads(data["downhaul_actual_daily_flight_mec"])
            backhaul_actual_daily_flight_mec = json.loads(data["backhaul_actual_daily_flight_mec"])

            med_first_res_solution_x = json.loads(data["med_first_res_solution_x"])
            med_first_res_solution_y = json.loads(data["med_first_res_solution_y"])

            med_last_res_solution_x = json.loads(data["med_last_res_solution_x"])
            med_last_res_solution_y = json.loads(data["med_last_res_solution_y"])

            down_end_to_back_start_days = json.loads(data["down_end_to_back_start_days"])
            med_last_downhaul_start_day = json.loads(data["med_last_downhaul_start_day"])
            med_last_backhaul_start_day = json.loads(data["med_last_backhaul_start_day"])
            items_grouping = int(json.loads(data["items_grouping"]))
            items_weight = json.loads(data["items_weight"])
            items_value = json.loads(data["items_value"])
            target_items_perc = json.loads(data["target_items_perc"])
        except Exception as e:
            return JsonResponse({"Error": "Wrong Data Entry - "+str(e)})

        try:
            # Making new and clean model - preventing over writing when several models run one after the other
            model.opt_model = model.gp.Model(name="Allocate MIP Model")
            model.downhaul_grouping_mfirst = model.flight_sol_grouping(
                downhaul_sol, downhaul_sol_res, downhaul_med_staff_reserve, downhaul_jed_staff_reserve, flight_capacity, True)
            print("Down no error")
            model.backhaul_grouping_mlast = model.flight_sol_grouping(
                backhaul_sol, backhaul_sol_res, backhaul_med_staff_reserve, backhaul_jed_staff_reserve, flight_capacity, False)
            # print(model.backhaul_grouping_mlast)
            # print(len(model.backhaul_grouping_mlast))

            model.visualize(
                med_first_prob,
                med_last_prob,
                med_first_prob.tot_days + down_end_to_back_start_days - 1,
                med_last_downhaul_start_day,
                med_last_backhaul_start_day,
                backhaul_actual_daily_flight_mec,
                downhaul_actual_daily_flight_mec,
                med_first_res_solution_x,
                med_first_res_solution_y,
                med_last_res_solution_x,
                med_last_res_solution_y,
                items_grouping,
                items_weight,
                items_value,
                target_items_perc,
                excel_output,
            )
        finally:
            optimal = 0
            inf = 0
            if model.opt_model.status == 2:
                optimal = 1
            if model.opt_model.status == 11:
                print("Model interrupted.")
            if model.solcnt > 0:
                import pandas as pd
                df = pd.DataFrame.from_dict(model.optimal_takhsis, orient='index')

                df.to_csv("./Output/Takhsis/allocate1_sol.csv")

            if model.opt_model.status == 3:
                inf = 1
                print("Error: Model is infeasible. Try lifting constraints.")

        return JsonResponse(buildResponse(optimal, inf))


def buildResponse(optimal=0, inf=0):
    if model.solcnt == 0:
        solution = "No results yet"
    else:
        med_first_table, med_last_table = makeSolList()

        solution = {
            'med_first_groups': {
                'table': json.dumps(med_first_table),
                'group_name': json.dumps(model.med_first_group_num),
                'total_duration': json.dumps(model.med_first_tot_duration),
            },
            'med_last_groups': {
                'table': json.dumps(med_last_table),
                'group_name': json.dumps(model.med_last_group_num),
                'total_duration': json.dumps(model.med_last_tot_duration),
            }
        }

    resp = {"runtime": model.runtime, "objbst": model.objbst, "objbnd": model.objbnd, "solcnt": model.solcnt,
            "gap": model.gap, "infeasible": str(inf), "optimal": str(optimal), "solution": solution}

    return resp


def makeSolList():

    med_first_table = []
    for d in range(len(med_first_prob.daily_flight_cap)):
        for i in range(len(model.final_mec_groups[d])):
            med_first_table.append([0]*len(items_weight))
            for w in range(len(items_weight)):
                med_first_table[-1][w] = int(model.optimal_takhsis[d, w, 1, i])

    med_last_table = []
    for d in range(med_last_prob.tot_days-med_last_prob.stay_duration+1):
        dd = d+len(med_first_prob.daily_flight_cap)
        for i in range(len(model.final_med2_groups[d])):
            med_last_table.append([0]*len(items_weight))
            for w in range(len(items_weight)):
                med_last_table[-1][w] = int(model.optimal_takhsis[dd, w, 1, i])

    # # print(model.optimal_takhsis)
    try:
        med_first_table_bin3 = []
        for d in range(len(med_first_prob.daily_flight_cap)):
            for i in range(len(model.downhaul_grouping_mfirst[d, :][model.downhaul_grouping_mfirst[d, :].nonzero()])):
                med_first_table_bin3.append([0]*len(items_weight))
                for w in range(len(items_weight)):
                    med_first_table_bin3[-1][w] = int(model.optimal_takhsis[d, w, 2, i])

    except:
        print("Error")

    # med_first_table_bin1 = []
    # for d in range(len(med_first_prob.daily_flight_cap)):
    #     for i in range(len(model.final_mec_groups[d])):
    #         med_first_table_bin1.append([0]*len(items_weight))
    #         for w in range(len(items_weight)):
    #             med_first_table_bin1[-1][w] = int(model.optimal_takhsis[d, w, 0, i])

    # print("med_first_table_bin1")
    # print(med_first_table_bin1)

    # print("med_first_table")
    # print(len(med_first_table))
    # print(len(med_first_table[0]))
    # print(med_first_table)

    return med_first_table, med_last_table


class AllocExcelView(APIView):
    def get(self, request, *args, **kwargs):
        global excel_output

        if model.solcnt == 0:
            return HttpResponse("No Results Yet")

        return excel_output
