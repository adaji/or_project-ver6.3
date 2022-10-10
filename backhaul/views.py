# from distutils.command.build import build
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render

# Third party imports
from rest_framework.response import Response
from rest_framework.views import APIView
import json

# My Optimization libraries
from . import flight_model4_param_back_toler as flight_model
import numpy as np

# Global var
prob = None


class BackhaulView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            if request.query_params.get('terminate') == "1":
                print('Terminating by Get')
                flight_model.opt_model.terminate()
        except:
            print("Provide terminate parameter in get request")
            # return JsonResponse({"Error": "Wrong Request"})

        return JsonResponse(buildResponse())

    def post(self, request, *args, **kwargs):
        data = request.data
        global prob
        # Serialize
        problem_names = ['downhaul_sol', 'downhaul_sol_res', 'downhaul_tot_days', 'downhaul_med_staff_reserve', 'downhaul_jed_staff_reserve', 'down_end_to_back_start_days', 'tot_days', 'days1', 'days2_start', 'tot_pass', 'med_staff_reserve',
                         'jed_staff_reserve', 'both_stations', 'station_passengers', 'flight_capacity', 'flight_capacity_toler', 'flight_cap_downhaul', 'airline_daily_max', 'airline_total_share', 'airline_total_share_tolerance',
                         'total_daily_slot_limit', 'daily_pass_limit_min', 'daily_pass_limit_max', 'daily_slot_cap_small_cities', 'daily_slot_cap_mid_cities']

        constraint_names = ['unavailable_flights_in_routes',
                            'flights_daily_cap', 'glued_flights', 'small_stations_routes', 'mid_stations_routes']
        try:
            prob_args = [json.loads(data[x]) for x in problem_names]
            constr_args = [json.loads(data[x]) for x in constraint_names]

            prob = flight_model.Problem(*prob_args)
            constr = flight_model.Constraints(*constr_args)
        except:
            return JsonResponse({"Error": "Wrong Data Entry"})

        try:
            # Making new and clean model - preventing over writing when several models run one after the other
            flight_model.opt_model = flight_model.gp.Model(name="Backhaul Flight MIP Model")
            flight_model.create_flight_model(prob, constr)
        finally:
            optimal = 0
            inf = 0
            if flight_model.opt_model.status == 2:
                optimal = 1
            if flight_model.opt_model.status == 11:
                print("Model interrupted.")
            if flight_model.solcnt > 0:
                sol_x, sol_xres, _, _, _, _ = makeSolList()
                flight_model.opt_model.write("./Output/Parvaz/backhaul_solution.sol")
                flight_model.visualize(prob, sol_x, sol_xres,
                                       './Output/Parvaz/backhaul_solution.xlsx')
            if flight_model.opt_model.status == 3:
                inf = 1
                print("Error: Model is infeasible. Try lifting constraints.")

        return JsonResponse(buildResponse(optimal, inf, final=True))


def buildResponse(optimal=0, inf=0, final=False):
    if flight_model.solcnt == 0:
        solution = 'No results yet'
    else:
        (
            sol_x,
            sol_xres,
            daily_flight_med,
            daily_flight_toler_med,
            daily_flight_mec,
            daily_flight_toler_mec
        ) = makeSolList(final)

        actual_daily_flight_med = []
        actual_daily_flight_mec = []
        if daily_flight_med != []:
            actual_daily_flight_med = [daily_flight_med[i] - daily_flight_toler_med[i]
                                       for i in range(len(daily_flight_med))]
        if daily_flight_mec != []:
            actual_daily_flight_mec = [daily_flight_mec[i] - daily_flight_toler_mec[i]
                                       for i in range(len(daily_flight_mec))]
        solution = {
            'x': json.dumps(sol_x.tolist()),
            'xresidual': json.dumps(sol_xres.tolist()),

            'daily_flight_mec': json.dumps(daily_flight_med),
            'daily_flight_toler_mec': json.dumps(daily_flight_toler_med),
            'actual_daily_flight_mec': json.dumps(actual_daily_flight_med),

            'daily_flight_med': json.dumps(daily_flight_mec),
            'daily_flight_toler_med': json.dumps(daily_flight_toler_mec),
            'actual_daily_flight_med': json.dumps(actual_daily_flight_mec),

            'total_pass_mec': json.dumps(sum(actual_daily_flight_med)),
            'total_pass_med': json.dumps(sum(actual_daily_flight_mec)),
            'med_last_backhaul_start_day': json.dumps(flight_model.med_last_backhaul_start_day)
        }

    resp = {"runtime": flight_model.runtime, "objbst": flight_model.objbst, "objbnd": flight_model.objbnd,
            "solcnt": flight_model.solcnt, "gap": flight_model.gap,  "infeasible": str(inf), "optimal": str(optimal), "solution": solution}

    return resp


def makeSolList(final=False):
    dim1 = list(flight_model.x_vars)[-1][0]+1
    dim2 = list(flight_model.x_vars)[-1][1]+1
    dim3 = list(flight_model.x_vars)[-1][2]+1

    sol_x = np.zeros((dim1, dim2, dim3), dtype=int)
    for f in range(dim1):
        for d in range(dim2):
            for r in range(dim3):
                sol_x[f, d, r] = flight_model.x_vars[(f, d, r)]

    sol_xres = np.zeros((dim1, dim2, dim3), dtype=int)
    for f in range(dim1):
        for d in range(dim2):
            for r in range(dim3):
                sol_xres[f, d, r] = flight_model.xres_vars[(f, d, r)]

    daily_flight_med = []
    daily_flight_mec = []
    daily_flight_toler_med = []
    daily_flight_toler_mec = []
    if final:
        global prob
        # Daily Flight Reserve for Haj Organization staff
        reserve = np.zeros((dim2, dim3), dtype=int)
        # Filling Tehran-Med & Tehran-Jed daily reserves
        reserve[:, 0] = prob.jed_staff_reserve
        reserve[:, int(dim3/2)] = prob.med_staff_reserve

        # Making daily flight cap and lost seat
        flight_capacity_raw = []  # will be [233, 250, 270, 290, 300, 430]
        [[flight_capacity_raw.append(y) for y in x] for x in prob.flight_capacity]
        # Medinah Calculation
        for d in range(sol_x.shape[1]):
            daily_cap = 0
            for f in range(sol_x.shape[0]):
                for r in range(len(prob.station_passengers)):
                    daily_cap += sol_x[f, d, r] * \
                        (flight_capacity_raw[f] - reserve[d, r]) - sol_xres[f, d, r]
            if daily_cap == 0:
                continue
            daily_flight_med.append(int(daily_cap))
        # Mecca Calculation
        for d in range(sol_x.shape[1]):
            daily_cap = 0
            for f in range(sol_x.shape[0]):
                for r in range(len(prob.station_passengers), 2*len(prob.station_passengers)):
                    daily_cap += sol_x[f, d, r] * \
                        (flight_capacity_raw[f] - reserve[d, r]) - sol_xres[f, d, r]
            if daily_cap == 0:
                continue
            daily_flight_mec.append(int(daily_cap))

        # Making Lost Seats
        entry_days = len(daily_flight_med)

        lost_seats_med = np.zeros(prob.tot_days)
        lost_seats_mec = np.zeros(prob.tot_days)
        lost_seats_fr = np.zeros((len(flight_capacity_raw), 2*len(prob.station_passengers)))
        extra_seats_in_routes = np.zeros(len(prob.station_passengers), dtype=int)
        extra_seats_in_routes_flight_capacity_weight_med = np.zeros(
            len(prob.station_passengers), dtype=int)
        extra_seats_in_routes_flight_capacity_weight_mec = np.zeros(
            len(prob.station_passengers), dtype=int)
        extra_seats_in_routes_days_med = [[]
                                          for i in range(len(prob.station_passengers))]
        extra_seats_in_routes_days_mec = [[]
                                          for i in range(len(prob.station_passengers))]

        last_total_pass_in_route = 0
        for s in range(len(prob.station_passengers)):
            for c in range(2):
                r = 2*s+c
                i = s+c*len(prob.station_passengers)

                total_flights_in_route = np.zeros(len(flight_capacity_raw))
                for d in range(entry_days):
                    tot = 0
                    for f in range(len(flight_capacity_raw)):
                        tot += sol_x[f, d, i]
                        total_flights_in_route[f] += sol_x[f, d, i]

                total_pass_in_route = 0
                for f in range(len(flight_capacity_raw)):
                    total_pass_in_route += total_flights_in_route[f] * \
                        flight_capacity_raw[f] - np.sum(sol_xres[f, :, i])

                if c == 1:
                    extra_seats = total_pass_in_route + \
                        last_total_pass_in_route - prob.station_passengers[s]
                    if extra_seats > 0:
                        extra_seats_in_routes[s] = extra_seats

                last_total_pass_in_route = total_pass_in_route

        # NEW: To remove staff reserve from extra seats
        extra_seats_in_routes[0] = 0
        # Lost Seat visualization

        for s in range(len(prob.station_passengers)):
            for c in range(2):
                i = s+c*len(prob.station_passengers)
                for d in range(prob.tot_days):
                    for f in range(len(flight_capacity_raw)):
                        if extra_seats_in_routes[s] > 0 and sol_x[f, d, i] > 0:
                            if c == 0:
                                if extra_seats_in_routes_days_med[s].count(d) == 0:
                                    extra_seats_in_routes_days_med[s].append(d)
                                extra_seats_in_routes_flight_capacity_weight_med[
                                    s] += flight_capacity_raw[f]*sol_x[f, d, i]-sol_xres[f, d, i]
                            else:
                                if extra_seats_in_routes_days_mec[s].count(d) == 0:
                                    extra_seats_in_routes_days_mec[s].append(d)
                                extra_seats_in_routes_flight_capacity_weight_mec[
                                    s] += flight_capacity_raw[f]*sol_x[f, d, i]-sol_xres[f, d, i]

        for s in range(len(prob.station_passengers)):
            if extra_seats_in_routes[s] > 0:
                i = s
                # if c == 0:
                for d in extra_seats_in_routes_days_med[s]:
                    w = 0
                    for f in range(len(flight_capacity_raw)):
                        if sol_x[f, d, i] > 0:
                            w += sol_x[f, d, i]*flight_capacity_raw[f]-sol_xres[f, d, i]
                            # Flight_Route Lost Seats
                            lost_seats_fr[f, i] += extra_seats_in_routes[s] / \
                                (extra_seats_in_routes_flight_capacity_weight_med[s] +
                                 extra_seats_in_routes_flight_capacity_weight_mec[s]) * (sol_x[f, d, i]*flight_capacity_raw[f]-sol_xres[f, d, i])

                    lost_seats_med[d] += extra_seats_in_routes[s] / \
                        (extra_seats_in_routes_flight_capacity_weight_med[s] +
                         extra_seats_in_routes_flight_capacity_weight_mec[s]) * w

                i = s+len(prob.station_passengers)
                for d in extra_seats_in_routes_days_mec[s]:
                    w = 0
                    for f in range(len(flight_capacity_raw)):
                        if sol_x[f, d, i] > 0:
                            w += sol_x[f, d, i]*flight_capacity_raw[f]-sol_xres[f, d, i]
                            # Flight_Route Lost Seats
                            lost_seats_fr[f, i] += extra_seats_in_routes[s] / \
                                (extra_seats_in_routes_flight_capacity_weight_med[s] +
                                 extra_seats_in_routes_flight_capacity_weight_mec[s]) * (sol_x[f, d, i]*flight_capacity_raw[f]-sol_xres[f, d, i])

                    lost_seats_mec[d] += extra_seats_in_routes[s] / \
                        (extra_seats_in_routes_flight_capacity_weight_med[s] +
                         extra_seats_in_routes_flight_capacity_weight_mec[s]) * w

        daily_flight_toler_med = lost_seats_med.astype(int).tolist()
        daily_flight_toler_mec = lost_seats_mec.astype(int).tolist()

    return sol_x, sol_xres, daily_flight_med, daily_flight_toler_med, daily_flight_mec, daily_flight_toler_mec


class BackhaulExcelView(APIView):
    def get(self, request, *args, **kwargs):
        response = HttpResponse(
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = "attachment; filename=Backhaul_Solution.xlsx"

        if flight_model.solcnt >= 0:
            sol_x, sol_xres, _, _, _, _ = makeSolList()
            flight_model.visualize(prob, sol_x, sol_xres, response)
        else:
            return HttpResponse("No Results Yet")

        return response
