from distutils.command.build import build
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render

# Third party imports
from rest_framework.response import Response
from rest_framework.views import APIView
import json

# My Optimization libraries
from . import res_model_45k_newobj as res_model
import numpy as np

# Global var
prob = None
prob_args = []


class ResidenceView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            if request.query_params.get('terminate') == "1":
                print('Terminating by Get')
                res_model.opt_model.terminate()
        except:
            print("Provide terminate parameter in get request")
            # return JsonResponse({"Error": "Wrong Request"})

        return JsonResponse(buildResponse())

    def post(self, request, *args, **kwargs):
        data = request.data
        global prob
        global prob_args
        # Serialize
        problem_names = ['med_first', 'daily_flight_med', 'station_passengers',
                         'stay_duration', 'hotel_price', 'hotel_capacity', 'hotel_availability']

        constraint_names = ['min_hotel_fill', 'daily_hotel_entry_cap', 'paying_early_days',
                            'paying_last_days', 'carevan_size_mult', 'carevan_size_res', 'remove_daily_mult_above_1']
        try:
            prob_args = [json.loads(data[x]) for x in problem_names]
            constr_args = [json.loads(data[x]) for x in constraint_names]

            prob = res_model.Problem(*prob_args)
            constr = res_model.Constraints(*constr_args)
        except Exception as e:
            return JsonResponse({"Error": "Wrong Data Entry: "+str(e)})

        try:
            # Making new and clean model - preventing over writing when several models run one after the other
            res_model.opt_model = res_model.gp.Model(name="Residenece MIP Model")
            res_model.create_res_model(prob, constr)
        finally:
            optimal = 0
            inf = 0
            if res_model.opt_model.status == 2:
                optimal = 1
            if res_model.opt_model.status == 11:
                print("Model interrupted.")
            if res_model.solcnt > 0:
                sol_x, sol_xres = makeSolList()
                if prob.med_first == 1:
                    res_model.opt_model.write("./Output/Eskan/res_plan_med_first.sol")
                    path = './Output/Eskan/res_plan_med_first.xlsx'
                else:
                    res_model.opt_model.write("./Output/Eskan/res_plan_med_last.sol")
                    path = './Output/Eskan/res_plan_med_last.xlsx'
                res_model.visualize(prob, sol_x, sol_xres, path)
            if res_model.opt_model.status == 3:
                inf = 1
                print("Error: Model is infeasible. Try lifting constraints.")

        return JsonResponse(buildResponse(optimal, inf))


def buildResponse(optimal=0, inf=0):
    if res_model.solcnt == 0:
        solution = "No results yet"
    else:
        sol_x, sol_y = makeSolList()

        solution = {
            'x': json.dumps(sol_x.tolist()),
            'y': json.dumps(sol_y.tolist()),
        }

    resp = {"runtime": res_model.runtime, "objbst": res_model.objbst, "objbnd": res_model.objbnd, "solcnt": res_model.solcnt,
            "gap": res_model.gap, "infeasible": str(inf), "optimal": str(optimal), "solution": solution}

    return resp


def makeSolList():
    dim1 = list(res_model.x_vars)[-1][0]+1
    dim2 = list(res_model.x_vars)[-1][1]+1
    sol_x = np.zeros((dim1, dim2), dtype=int)
    for h in range(dim1):
        for d in range(dim2):
            sol_x[h, d] = res_model.x_vars[(h, d)]

    dim1 = list(res_model.y_vars)[-1][0]+1
    dim2 = list(res_model.y_vars)[-1][1]+1
    sol_y = np.zeros((dim1, dim2), dtype=int)
    for h in range(dim1):
        for d in range(dim2):
            sol_y[h, d] = res_model.y_vars[(h, d)]

    return sol_x, sol_y


class ResExcelView(APIView):
    def get(self, request, *args, **kwargs):
        response = HttpResponse(
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        if prob_args[0] == 1:
            response['Content-Disposition'] = "attachment; filename=Res_Plan_Med_First.xlsx"
        else:
            response['Content-Disposition'] = "attachment; filename=Res_Plan_Med_Last.xlsx"

        if res_model.solcnt > 0:
            sol_x, sol_xres, = makeSolList()
            res_model.visualize(prob, sol_x, sol_xres, response)
        else:
            return HttpResponse("No Results Yet")

        return response
