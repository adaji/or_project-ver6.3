import gurobipy as gp
from gurobipy import GRB
import xlsxwriter as xw
from xlsxwriter.utility import xl_rowcol_to_cell
import numpy as np

opt_model = gp.Model(name="Backhaul Flight MIP Model")
x_vars = []
xres_vars = []
y_vars = []
route_priority = np.zeros(0)
med_last_backhaul_start_day = 0

routes_num = 0
flight_num = 0

runtime = 0
objbst = 0
objbnd = 0
solcnt = 0
gap = 0.0

# BACK HAUL PROBLEM

# Important Notes
# Routes are like [BOTH_CITIES, DAYS1_CITIES, BOTH_CITIES, DAYS2_CITIES]
# Route 0  must be TEHRAN-MEDINAH
# Airline 0 must be IranAir


class Problem:
    def __init__(self,
                 downhaul_sol=[],
                 downhaul_sol_res=[],
                 downhaul_tot_days=0,
                 downhaul_med_staff_reserve=[],
                 downhaul_jed_staff_reserve=[],
                 down_end_to_back_start_days=0,
                 tot_days=0,
                 days1=0,
                 days2_start=0,
                 tot_pass=0,
                 med_staff_reserve=[],
                 jed_staff_reserve=[],
                 both_stations=[],
                 station_passengers=[],
                 flight_capacity=[],
                 flight_capacity_toler=[],
                 flight_cap_downhaul=[],
                 airline_daily_max=[],
                 airline_total_share=[],
                 airline_total_share_tolerance=0,
                 total_daily_slot_limit=[],
                 daily_pass_limit_min=0,
                 daily_pass_limit_max=0,
                 daily_slot_cap_small_cities=0,
                 daily_slot_cap_mid_cities=0,
                 ) -> None:
        self.downhaul_sol = downhaul_sol
        self.downhaul_sol_res = downhaul_sol_res
        self.downhaul_tot_days = downhaul_tot_days
        self.downhaul_med_staff_reserve = downhaul_med_staff_reserve
        self.downhaul_jed_staff_reserve = downhaul_jed_staff_reserve
        self.down_end_to_back_start_days = down_end_to_back_start_days
        self.tot_days = tot_days
        self.days1 = days1
        self.days2_start = days2_start
        self.tot_pass = tot_pass
        self.med_staff_reserve = med_staff_reserve
        self.jed_staff_reserve = jed_staff_reserve
        self.both_stations = both_stations
        self.station_passengers = station_passengers
        self.flight_capacity = flight_capacity
        self.flight_capacity_toler = flight_capacity_toler
        self.flight_cap_downhaul = flight_cap_downhaul
        self.airline_daily_max = airline_daily_max
        self.airline_total_share = airline_total_share
        self.airline_total_share_tolerance = airline_total_share_tolerance
        self.total_daily_slot_limit = total_daily_slot_limit
        self.daily_pass_limit_min = daily_pass_limit_min
        self.daily_pass_limit_max = daily_pass_limit_max
        self.daily_slot_cap_small_cities = daily_slot_cap_small_cities
        self.daily_slot_cap_mid_cities = daily_slot_cap_mid_cities


class Constraints:
    def __init__(self,
                 unavailable_flights_in_routes={},
                 flights_daily_cap={},
                 glued_flights=[],
                 small_stations_routes=[],
                 mid_stations_routes=[],
                 ) -> None:
        self.unavailable_flights_in_routes = unavailable_flights_in_routes
        self.flights_daily_cap = flights_daily_cap
        self.glued_flights = glued_flights
        self.small_stations_routes = small_stations_routes
        self.mid_stations_routes = mid_stations_routes


def create_flight_model(prob=Problem(), constr=Constraints()):
    # Preprocessing data
    global flight_num
    flight_num = sum([len(e) for e in prob.flight_capacity])
    airline_num = len(prob.flight_capacity)
    all_flights = range(flight_num)
    flight_capacity_raw = []  # will be [233, 250, 270, 290, 300, 430]
    [[flight_capacity_raw.append(y) for y in x] for x in prob.flight_capacity]
    all_flights_al = []  # will be [[0,1],[2,3],[4,5]]
    al_count = 0
    for a in prob.flight_capacity:
        l = []
        for x, y in enumerate(a):
            l.append(al_count+x)
        all_flights_al.append(l)
        al_count += len(a)
    all_days = range(prob.tot_days)
    overlap_days = range(prob.days2_start-1, prob.days1)

    both_route_num = len(prob.both_stations)
    global routes_num
    routes_num = 2*len(prob.station_passengers)
    # routes_num = 2*both_route_num + days1_route_num + days2_route_num

    stations_num = len(prob.station_passengers)
    all_stations = range(stations_num)

    both_routes = [[x, x+stations_num]
                   for x in range(both_route_num)]
    both_routes_raw = []  # will be [233, 250, 270, 290, 300, 430]
    [[both_routes_raw.append(y) for y in x] for x in both_routes]
    all_days1_routes = range(stations_num)
    all_days2_routes = range(stations_num, routes_num)

    all_routes = range(routes_num)

    # Daily Flight Reserve for Haj Organization staff
    reserve = np.zeros((prob.tot_days, routes_num), dtype=int)
    # Filling Tehran-Med & Tehran-Jed daily reserves
    reserve[:, 0] = prob.jed_staff_reserve
    reserve[:, stations_num] = prob.med_staff_reserve

    # Same above for Downhaul:
    # Daily Flight Reserve for Haj Organization staff
    downhaul_reserve = np.zeros((prob.downhaul_tot_days, routes_num), dtype=int)
    # Filling Tehran-Med & Tehran-Jed daily reserves
    downhaul_reserve[:, 0] = prob.downhaul_med_staff_reserve
    downhaul_reserve[:, stations_num] = prob.downhaul_jed_staff_reserve

    # Get Data from gloal var instead of .sol file
    # with open(constr.downhaul_sol) as f:
    #     content = f.read()

    #     x0 = re.findall(
    #         r'X\((\d+)_(\d+)_(\d+)\) (\d+)', content)

    #     X0 = np.zeros(
    #         (flight_num, prob.downhaul_tot_days, routes_num), dtype=int)
    #     for x in x0:
    #         # Remove -1 if using optimal solution - -1 just for X0
    #         X0[int(x[0]), int(x[1]), int(x[2])] = int(x[3])

    #     X0_res = re.findall(
    #         r'XRES\((\d+)_(\d+)_(\d+)\) (\d+)', content)
    #     X0Res = np.zeros(
    #         (flight_num, prob.downhaul_tot_days, routes_num), dtype=int)
    #     for x in X0_res:
    #         X0Res[int(x[0]), int(x[1]), int(x[2])] = int(x[3])
    # Getting solution from global var instead of above
    X0 = np.array(prob.downhaul_sol)
    X0Res = np.array(prob.downhaul_sol_res)

    down_start_day = np.zeros(routes_num) - 1
    for r in range(routes_num):
        for d in range(prob.downhaul_tot_days):
            for f in range(len(prob.flight_cap_downhaul)):
                if X0[f, d, r] > 0:
                    down_start_day[r] = d
                    break
            if down_start_day[r] >= 0:
                break

    # Calculate priority matrix
    deployed_routes_passengers = np.zeros(routes_num)
    global route_priority
    route_priority = np.zeros(routes_num)
    for d in range(prob.downhaul_tot_days):
        for r in all_routes:
            day_route_f = 0
            for f in range(len(prob.flight_cap_downhaul)):
                day_route_f += X0[f, d, r]
                deployed_routes_passengers[r] += \
                    X0[f, d, r]*(prob.flight_cap_downhaul[f] - downhaul_reserve[d, r]) - X0Res[f, d, r]

            if day_route_f > 0 and route_priority[r] == 0:
                route_priority[r] = d+1
                # print("Route %i : Day %i" % (r, d+1))

    # deployed_routes_passengers[0] += 1020
    # deployed_routes_passengers[stations_num] += 520

    # print(deployed_routes_passengers)
    # print("Priorities:")
    # print(route_priority)
    print("downhaul_reserve")
    print(downhaul_reserve.tolist())
    print("deployed_routes_passengers")
    print(deployed_routes_passengers.tolist())

    # Lost Seat in Downhaul calculation
    lost_seats_fr = np.zeros((flight_num, routes_num))
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

            total_flights_in_route = np.zeros(flight_num)
            for d in range(prob.downhaul_tot_days):
                tot = 0
                for f in range(flight_num):
                    tot += X0[f, d, i]
                    total_flights_in_route[f] += X0[f, d, i]

            total_pass_in_route = 0
            for f in range(flight_num):
                total_pass_in_route += total_flights_in_route[f] * \
                    prob.flight_cap_downhaul[f] - np.sum(X0Res[f, :, i])

            if c == 1:
                extra_seats = total_pass_in_route + \
                    last_total_pass_in_route - prob.station_passengers[s]
                if extra_seats > 0:
                    extra_seats_in_routes[s] = extra_seats

            last_total_pass_in_route = total_pass_in_route

    # NEW: To remove staff reserve from extra seats
    extra_seats_in_routes[0] = 0

    for s in range(len(prob.station_passengers)):
        for c in range(2):
            i = s+c*len(prob.station_passengers)
            for d in range(prob.downhaul_tot_days):
                for f in range(flight_num):
                    if extra_seats_in_routes[s] > 0 and X0[f, d, i] > 0:
                        if c == 0:
                            if extra_seats_in_routes_days_med[s].count(d) == 0:
                                extra_seats_in_routes_days_med[s].append(d)
                            extra_seats_in_routes_flight_capacity_weight_med[
                                s] += (prob.flight_cap_downhaul[f] - reserve[d, i])*X0[f, d, i]-X0Res[f, d, i]
                        else:
                            if extra_seats_in_routes_days_mec[s].count(d) == 0:
                                extra_seats_in_routes_days_mec[s].append(d)
                            extra_seats_in_routes_flight_capacity_weight_mec[
                                s] += (prob.flight_cap_downhaul[f] - reserve[d, i])*X0[f, d, i]-X0Res[f, d, i]

    for s in range(len(prob.station_passengers)):
        if extra_seats_in_routes[s] > 0:
            i = s
            # if c == 0:
            for d in extra_seats_in_routes_days_med[s]:
                w = 0
                for f in range(flight_num):
                    if X0[f, d, i] > 0:
                        w += X0[f, d, i] * \
                            (prob.flight_cap_downhaul[f] - reserve[d, i])-X0Res[f, d, i]
                        # Flight_Route Lost Seats
                        lost_seats_fr[f, i] += extra_seats_in_routes[s] / \
                            (extra_seats_in_routes_flight_capacity_weight_med[s] +
                             extra_seats_in_routes_flight_capacity_weight_mec[s]) * (X0[f, d, i]*(prob.flight_cap_downhaul[f] - reserve[d, i])-X0Res[f, d, i])

            i = s+len(prob.station_passengers)
            for d in extra_seats_in_routes_days_mec[s]:
                w = 0
                for f in range(flight_num):
                    if X0[f, d, i] > 0:
                        w += X0[f, d, i]*(flight_capacity_raw[f] - reserve[d, i])-X0Res[f, d, i]
                        # Flight_Route Lost Seats
                        lost_seats_fr[f, i] += extra_seats_in_routes[s] / \
                            (extra_seats_in_routes_flight_capacity_weight_med[s] +
                             extra_seats_in_routes_flight_capacity_weight_mec[s]) * (X0[f, d, i]*(prob.flight_cap_downhaul[f] - reserve[d, i])-X0Res[f, d, i])
    lost_by_route = np.sum(lost_seats_fr, axis=0).astype(int)

    # End Lost Seat in Downhaul calculation

    # Optimization Model
    x = {(f, d, r): opt_model.addVar(vtype=GRB.INTEGER, lb=0, ub=6, name='X(%i_%i_%i)' % (f, d, r))
         for f in all_flights for d in all_days for r in all_routes}

    xresidual = {(f, d, r): opt_model.addVar(vtype=GRB.INTEGER, lb=0, name='XRES(%i_%i_%i)' % (f, d, r))
                 for f in all_flights for d in all_days for r in all_routes}
    # Tehran residuals
    opt_model.addConstrs(xresidual[f, d, r] <= x[f, d, r] * prob.flight_capacity_toler[f]
                         for f in all_flights for d in all_days for r in all_routes)  # [0, stations_num])
    # Other stations residuals
    # opt_model.addConstrs(xresidual[f, d, r] <= x[f, d, r] * 5
    #                      for f in all_flights for d in all_days for r in list(set(all_routes)-set([0, stations_num])))

    y = {(d, r): opt_model.addVar(vtype=GRB.BINARY, name='Y(%i_%i)' % (d, r))
         for d in all_days for r in all_routes}

    opt_model.addConstrs(10000*y[d, r] >= gp.quicksum(x[f, d, r] for f in all_flights)
                         for d in all_days for r in all_routes)
    opt_model.addConstrs(y[d, r] <= gp.quicksum(x[f, d, r] for f in all_flights)
                         for d in all_days for r in all_routes)

    # All Passengers should be deployed
    print("lost_by_route")
    print(lost_by_route)
    # HANDL difference in backhaul by 7 people
    deployed_routes_passengers[stations_num] -= 7

    # If deployed route passengers is different than original passenger
    opt_model.addConstrs(gp.quicksum(x[f, d, r]*(flight_capacity_raw[f]-reserve[d, r])-xresidual[f, d, r] for f in all_flights for d in all_days)
                         == deployed_routes_passengers[r] - lost_by_route[r]
                         # /sum(deployed_routes_passengers)*prob.tot_pass
                         for r in all_routes)

    opt_model.addConstr(gp.quicksum(x[f, d, 0]*reserve[d, 0]
                        for f in all_flights for d in all_days) >= 1020)
    opt_model.addConstr(gp.quicksum(x[f, d, stations_num]*reserve[d, stations_num]
                        for f in all_flights for d in all_days) >= 520)
    # opt_model.addConstrs(gp.quicksum(x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r]
    #                                  for f in all_flights for d in all_days for r in [s, s+stations_num]) >= prob.station_passengers[s]
    #                      for s in all_stations)

    # Restrict Medina First & Last Routes
    opt_model.addConstrs(x[f, d, r] == 0
                         for f in all_flights for d in range(prob.days1, prob.tot_days) for r in all_days1_routes)

    opt_model.addConstrs(x[f, d, r] == 0
                         for f in all_flights for d in range(prob.days2_start-1) for r in all_days2_routes)

    # Flight-Route Restrictions
    opt_model.addConstrs(x[f, d, r] == 0
                         for f in constr.unavailable_flights_in_routes.keys() for d in all_days for r in constr.unavailable_flights_in_routes[f])

    # Tehran everyday, to both med jed
    opt_model.addConstrs(y[d, 0] == 1
                         for d in range(prob.days1))

    opt_model.addConstrs(y[d, stations_num] == 1
                         for d in range(prob.days1, prob.tot_days))

    opt_model.addConstrs(10000*y[d, stations_num] >= gp.quicksum(x[f, d, r] for f in all_flights for r in all_days2_routes)
                         for d in overlap_days)
    opt_model.addConstrs(y[d, stations_num] <= gp.quicksum(x[f, d, r] for f in all_flights for r in all_days2_routes)
                         for d in overlap_days)

    # Med-first & Med-last stations should not have overlap daily (All both ways except Tehran indice 0)-Def: Mashhad & Isfahan
    opt_model.addConstrs(y[d, r] + y[d, r+stations_num] <= 1
                         for r in list(set(prob.both_stations)-set([0])) for d in all_days)

    # Gluing station flights together
    w1 = {(d, r): opt_model.addVar(vtype=GRB.BINARY, name='W1(%i_%i)' % (d, r))
          for d in all_days for r in all_routes}
    w2 = {(d, r): opt_model.addVar(vtype=GRB.BINARY, name='W2(%i_%i)' % (d, r))
          for d in all_days for r in all_routes}
    z = {(d, r): opt_model.addVar(vtype=GRB.BINARY, name='Z(%i_%i)' % (d, r))
         for d in all_days for r in all_routes}

    opt_model.addConstrs(w1[d, r] == gp.any_(y[i, r] for i in range(d))
                         for d in all_days for r in all_routes)
    opt_model.addConstrs(w2[d, r] == gp.any_(y[i, r] for i in range(d+1, prob.tot_days))
                         for d in all_days for r in all_routes)
    opt_model.addConstrs((y[d, r] == 0) >> (z[d, r] >= w1[d, r] + w2[d, r] - 1)
                         for d in all_days for r in all_routes)

    obj1 = gp.quicksum(200000000*z[d, r]
                       for d in all_days for r in all_routes)  # 1000 weight

    # Removing 1 total flight each day on each route
    # Routes that have just flight doesnt need this criteria
    all_routes_aobve_1_flight = list(all_routes).copy()
    just_1_routes = []
    for s in range(stations_num):
        if prob.station_passengers[s] <= max(flight_capacity_raw):
            # print("1 Flight routes:")
            # print(s, stations_num+s)
            just_1_routes.append(s)
            just_1_routes.append(stations_num+s)
            all_routes_aobve_1_flight.remove(s)
            all_routes_aobve_1_flight.remove(stations_num+s)
            # print(all_routes_aobve_1_flight)
    # tehran can be 1 as well
    all_routes_aobve_1_flight.remove(0)
    all_routes_aobve_1_flight.remove(stations_num)

    # New: just 1 routes limit
    opt_model.addConstrs(gp.quicksum(x[f, d, r] for f in all_flights for d in all_days) <= 1
                         for r in just_1_routes)

    p = {(d, r): opt_model.addVar(vtype=GRB.BINARY, name='P(%i_%i)' % (d, r))
         for d in all_days for r in all_routes_aobve_1_flight}

    opt_model.addConstrs((y[d, r] == 1) >> (100000000*p[d, r] >= gp.quicksum(x[f, d, r] for f in all_flights)-y[d, r])
                         for d in all_days for r in all_routes_aobve_1_flight)
    opt_model.addConstrs((y[d, r] == 1) >> (p[d, r] <= gp.quicksum(x[f, d, r] for f in all_flights)-y[d, r])
                         for d in all_days for r in all_routes_aobve_1_flight)
    opt_model.addConstrs((y[d, r] == 0) >> (p[d, r] == 1)
                         for d in all_days for r in all_routes_aobve_1_flight)

    obj2 = gp.quicksum(200000000*(1-p[d, r])
                       for d in all_days for r in all_routes_aobve_1_flight)  # 5000 weight

    fr = {(f, r): opt_model.addVar(vtype=GRB.BINARY, name='FR(%i_%i)' % (f, r))
          for f in range(airline_num) for r in list(set(list(range(routes_num))) - set([0, stations_num]))}

    for al in range(airline_num):
        opt_model.addConstrs(10000*fr[al, r] >= gp.quicksum(x[f, d, r]
                                                            for f in all_flights_al[al] for d in all_days)
                             for r in list(set(list(range(routes_num))) - set([0, stations_num])))

        opt_model.addConstrs(fr[al, r] <= gp.quicksum(x[f, d, r]
                                                      for f in all_flights_al[al] for d in all_days)
                             for r in list(set(list(range(routes_num))) - set([0, stations_num])))

    # Restrict each station to have one airline except Tehran
    opt_model.addConstrs(gp.quicksum(fr[f, r] for f in range(airline_num)) <= 1
                         for r in list(set(list(range(routes_num))) - set([0, stations_num])))
    # Restrict other both way stations
    opt_model.addConstrs(fr[f, both_routes[s][0]] == fr[f, both_routes[s][1]]
                         for s in list(set(prob.both_stations)-set([0]))
                         for f in range(airline_num))

    # Airline daily cap
    opt_model.addConstrs(gp.quicksum(x[f, d, r] for f in all_flights_al[al] for r in all_routes) <= prob.airline_daily_max[al]
                         for al in range(airline_num) for d in all_days)

    # Daily passenger limit
    # Med Only
    opt_model.addConstrs(gp.quicksum(x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r] for f in all_flights for r in all_routes) <= prob.total_daily_slot_limit[0]
                         for d in range(prob.days2_start-1))
    # # Med & Jed Parallel
    opt_model.addConstrs(gp.quicksum(x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r] for f in all_flights for r in all_routes) <= prob.total_daily_slot_limit[1]
                         for d in overlap_days)
    # # Jed Only
    opt_model.addConstrs(gp.quicksum(x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r] for f in all_flights for r in all_routes) <= prob.total_daily_slot_limit[0]
                         for d in range(prob.days1, prob.tot_days))

    # Daily Med Out Limit
    opt_model.addConstrs(gp.quicksum(x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r] for f in all_flights for r in all_days2_routes) <= prob.daily_pass_limit_max[d]
                         for d in range(prob.days2_start-1, prob.tot_days))

    opt_model.addConstrs(gp.quicksum(x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r] for f in all_flights for r in all_days2_routes) >= prob.daily_pass_limit_min[d]
                         for d in range(prob.days2_start-1, prob.days1))

    # NEW: Define minimum duration - 11 for min 30 - not feasible -> 12 for min 29 feasible good
    for r in all_routes:
        if down_start_day[r] > prob.down_end_to_back_start_days:
            opt_model.addConstrs(y[d, r] == 0
                                 for d in range(int(down_start_day[r])-prob.down_end_to_back_start_days))

    # NEW: Define maximum duration - Not longer than 41 _ WORKS
    for r in all_routes:
        if down_start_day[r] >= 0:
            opt_model.addConstr(gp.quicksum(y[d, r] for d in range(
                min(int(down_start_day[r])+1, prob.tot_days))) >= 1)

    # Flights Daily Cap
    # NEW DEFINITION: flights_daily_cap= {CAP1: [[Flights Group1],[Flight Group 2],...], CAP2: [[8,9],[4]]}
    # Example: {4: [[8,9]]} or {4: [[5]]}
    for cap in constr.flights_daily_cap.keys():
        for flight_group in constr.flights_daily_cap[cap]:
            opt_model.addConstrs(gp.quicksum(x[f, d, r] for f in flight_group for r in all_routes) <= cap
                                 for d in all_days)

    obj3 = 0
    if constr.glued_flights != []:
        # # Gluing flights together-all (most important is saudi jambo 430)
        fy = {(f, d): opt_model.addVar(vtype=GRB.BINARY, name='FY1(%i_%i)' % (f, d))
              for f in range(len(constr.glued_flights)) for d in all_days}
        fw1 = {(f, d): opt_model.addVar(vtype=GRB.BINARY, name='FW1(%i_%i)' % (f, d))
               for f in range(len(constr.glued_flights)) for d in all_days}
        fw2 = {(f, d): opt_model.addVar(vtype=GRB.BINARY, name='FW2(%i_%i)' % (f, d))
               for f in range(len(constr.glued_flights)) for d in all_days}
        fz = {(f, d): opt_model.addVar(vtype=GRB.BINARY, name='FZ(%i_%i)' % (f, d))
              for f in range(len(constr.glued_flights)) for d in all_days}

        # FY means if flight f has any flights on day d
        opt_model.addConstrs(10000*fy[f, d] >= gp.quicksum(x[myf, d, r] for myf in constr.glued_flights[f] for r in all_routes)
                             for f in range(len(constr.glued_flights)) for d in all_days)
        opt_model.addConstrs(fy[f, d] <= gp.quicksum(x[myf, d, r] for myf in constr.glued_flights[f] for r in all_routes)
                             for f in range(len(constr.glued_flights)) for d in all_days)

        opt_model.addConstrs(fw1[f, d] == gp.any_(fy[f, i] for i in range(d))
                             for f in range(len(constr.glued_flights)) for d in all_days)
        opt_model.addConstrs(fw2[f, d] == gp.any_(fy[f, i] for i in range(d+1, prob.tot_days))
                             for f in range(len(constr.glued_flights)) for d in all_days)
        opt_model.addConstrs((fy[f, d] == 0) >> (fz[f, d] >= fw1[f, d] + fw2[f, d] - 1)
                             for f in range(len(constr.glued_flights)) for d in all_days)

        obj3 = gp.quicksum(
            200000000*fz[f, d] for f in range(len(constr.glued_flights)) for d in all_days)  # 1-100 weight

    # Airlines total passenger restrictions _ Only floor can be defined
    # opt_model.addConstr(gp.quicksum(x[f, d, r]*(flight_capacity[f] - reserve[d, r]) for f in range(2)
    #                                 for d in all_days for r in all_routes) <= (airline_total_share[0]+ATSAT+0.015)*tot_pass)
    opt_model.addConstrs(gp.quicksum(x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r] for f in all_flights_al[al]
                                     for d in all_days for r in all_routes) >= (prob.airline_total_share[al]-prob.airline_total_share_tolerance)*prob.tot_pass
                         for al in range(airline_num))

    # Small Stations Daily limit
    opt_model.addConstrs(gp.quicksum(x[f, d, r] for f in all_flights) <= prob.daily_slot_cap_small_cities
                         for d in all_days
                         for r in constr.small_stations_routes)

    opt_model.addConstrs(gp.quicksum(x[f, d, r] for f in all_flights) <= prob.daily_slot_cap_mid_cities
                         for d in all_days
                         for r in constr.mid_stations_routes)

    # Routes (Stations) priority
    # Priorities based on Down Haul
    priority1 = gp.quicksum((x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r])/50000*(routes_num*(1-route_priority[r]/(sum(route_priority))))*(d+prob.tot_days)
                            for f in all_flights for d in all_days for r in all_routes)  # 100 weight

    # Put all flights as early as possible
    # priority2 = gp.quicksum(y[d, r]*(d-route_priority[r])
    #                         for d in all_days for r in all_routes)
    # priority2 = gp.quicksum((x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r])*0.01*(prob.tot_days-d)
    #                         for f in all_flights for d in all_days for r in all_days2_routes)

    # # opt_model.addConstrs(y[d, r]*(43+d-route_priority[r])
    # #                      >= 17 for d in all_days for r in list(set(all_days2_routes) - set([0, all_days1_route_num])))
    # opt_model.addConstrs(y[d, r]*(43+d-route_priority[r])
    #                      <= 35 for d in all_days for r in list(set(all_days2_routes) - set([0, all_days1_route_num])))

    # # opt_model.addConstrs(y[d, r]*(43+d-route_priority[r])
    # #                      >= 17 for d in all_days for r in list(set(all_days1_routes) - set([0, all_days1_route_num])))
    # opt_model.addConstrs(y[d, r]*(43+d-route_priority[r])
    #                      <= 42 for d in all_days for r in list(set(all_days1_routes) - set([0, all_days1_route_num])))

    # Objective
    obj4 = 1.6*gp.quicksum(x[f, d, r]*(flight_capacity_raw[f] - reserve[d, r])-xresidual[f, d, r]
                           for f in all_flights for d in all_days for r in all_routes)  # 2 weight

    print('Start Optimization...')
    opt_model.ModelSense = GRB.MINIMIZE
    opt_model.setObjective(obj1+obj2+obj3+obj4 + priority1)  # p2 ommited

    opt_model.write("./Output/Parvaz/backhaul_model.lp")

    # Model Parameters
    opt_model.Params.timelimit = 10000
    opt_model.Params.integralityfocus = 1
    # opt_model.Params.mipfocus = 1
    # opt_model.Params.seed = 12424

    def getSol(model, where):
        if where == GRB.Callback.MIPSOL:
            global x_vars
            global xres_vars
            global y_vars
            global solcnt

            x_vars = model.cbGetSolution(x)
            xres_vars = model.cbGetSolution(xresidual)
            y_vars = model.cbGetSolution(y)

            solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
            objective_value = 0

            for f in range(flight_num):
                all_same_flight = 0
                for d in range(prob.tot_days):
                    for r in range(routes_num):
                        all_same_flight += x_vars[f, d, r]
                objective_value += all_same_flight*flight_capacity_raw[f]
            print("\t\t\tObj: %i" % (objective_value))

        elif where == GRB.Callback.MIP:
            global runtime
            global objbst
            global objbnd
            global gap

            runtime = model.cbGet(GRB.Callback.RUNTIME)
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            gap = (objbst-objbnd)/objbst

    opt_model.optimize(getSol)


def visualize(prob, sol_x, sol_xres, response):
    flight_count = sum([len(e) for e in prob.flight_capacity])
    day_count = prob.tot_days

    route_count = 2*len(prob.station_passengers)

    flight_capacity = []  # will be [233, 250, 270, 290, 300, 430]
    [[flight_capacity.append(y) for y in x] for x in prob.flight_capacity]

    # Daily Flight Reserve for Haj Organization staff
    reserve = np.zeros((prob.tot_days, 2*len(prob.station_passengers)), dtype=int)
    # Filling Tehran-Med & Tehran-Jed daily reserves
    reserve[:, 0] = prob.jed_staff_reserve
    reserve[:, len(prob.station_passengers)] = prob.med_staff_reserve

    route_descr = ['IKA', 'MHD', 'IFN', 'AWZ', 'SRY', 'AZD', 'KER', 'RAS', 'ADU', 'TBZ', 'JWN',
                   'ZAH', 'BND', 'XBJ', 'OMH', 'BUZ', 'KSH', 'GBT', 'HDM', 'SYZ']
    route_descr = ['JED-'+x for x in route_descr] + ['MED-'+x for x in route_descr]

    lost_seats_med = np.zeros(day_count)
    lost_seats_mec = np.zeros(day_count)
    lost_seats_fr = np.zeros((flight_count, route_count))
    extra_seats_in_routes = np.zeros(len(prob.station_passengers), dtype=int)
    extra_seats_in_routes_flight_capacity_weight_med = np.zeros(
        len(prob.station_passengers), dtype=int)
    extra_seats_in_routes_flight_capacity_weight_mec = np.zeros(
        len(prob.station_passengers), dtype=int)
    extra_seats_in_routes_days_med = [[]
                                      for i in range(len(prob.station_passengers))]
    extra_seats_in_routes_days_mec = [[]
                                      for i in range(len(prob.station_passengers))]

    # Get Data from gloal var instead of .sol file
    # with open('backhaul.sol') as f:
    #     content = f.read()

    #     entry = re.findall(
    #         r'X\((\d+)_(\d+)_(\d+)\) (\d+)', content)

    #     X = np.zeros(
    #         (flight_count, day_count, route_count))
    #     for x in entry:
    #         # Remove -1 if using optimal solution - -1 just for X0
    #         X[int(x[0]), int(x[1]), int(x[2])] = int(x[3])
    #     entry_res = re.findall(
    #         r'XRES\((\d+)_(\d+)_(\d+)\) (\d+)', content)
    #     XRes = np.zeros(
    #         (flight_count, day_count, route_count), dtype=int)
    #     for x in entry_res:
    #         XRes[int(x[0]), int(x[1]), int(x[2])] = int(x[3])
    X = sol_x
    XRes = sol_xres

    # Get Data from gloal var instead of .sol file
    # with open('downhaul.sol') as f:
    #     content = f.read()

    #     entry = re.findall(
    #         r'X\((\d+)_(\d+)_(\d+)\) (\d+)', content)

    #     X0 = np.zeros(
    #         (len(prob.flight_cap_downhaul), prob.downhaul_tot_days, route_count), dtype=int)
    #     for x in entry:
    #         # Remove -1 if using optimal solution - -1 just for X0
    #         X0[int(x[0]), int(x[1]), int(x[2])] = int(x[3])
    #     X0_res = re.findall(
    #         r'XRES\((\d+)_(\d+)_(\d+)\) (\d+)', content)
    #     X0Res = np.zeros(
    #         (len(prob.flight_cap_downhaul), prob.downhaul_tot_days, route_count), dtype=int)
    #     for x in X0_res:
    #         X0Res[int(x[0]), int(x[1]), int(x[2])] = int(x[3])
    X0 = np.array(prob.downhaul_sol)
    X0Res = np.array(prob.downhaul_sol_res)

    # Calculating each route's travel duration
    max_duration = np.zeros(route_count)
    for r in range(route_count):
        down_start_day = -1
        for d in range(prob.downhaul_tot_days):
            for f in range(len(prob.flight_cap_downhaul)):
                if X0[f, d, r] > 0:
                    down_start_day = d+1
                    break
            if down_start_day > 0:
                break

        back_start_day = -1
        for d in range(day_count):
            for f in range(flight_count):
                if X[f, d, r] > 0:
                    back_start_day = d+1
                    break
            if back_start_day > 0:
                break

        if down_start_day != -1 and back_start_day != -1:
            max_duration[r] = back_start_day + 41 - down_start_day

    # Calculating objective function value for current solution
    objective_value = 0

    for f in range(flight_count):
        all_same_flight = 0
        for d in range(day_count):
            for r in range(route_count):
                all_same_flight += X[f, d, r]
        objective_value += all_same_flight*flight_capacity[f]

    print("Objective value for current solution: %i" % (objective_value))

    # Creating spreadsheet
    workbook = xw.Workbook(response, {'in_memory': True})
    ws = workbook.add_worksheet('Flight Based')  # ws means worksheet

    # Before writing values, define formatting
    blank_fmt = workbook.add_format({'bg_color': '#FFFFFF'})
    obj_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#003296', 'font_color': '#FFFFFF', 'bold': True})
    obj2_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#003296', 'font_color': '#FF2600'})
    obj3_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#003296', 'font_color': '#BFBFBF'})
    tot_entry_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#ED7D31', 'align': 'center', 'bold': True})
    tot_entry_title_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#ED7D31', 'align': 'left', 'bold': True})
    tot_entry_diff_fmt = workbook.add_format(
        {'italic': True, 'bg_color': '#F4B084', 'align': 'center', 'bold': True})
    tot_entry_diff_tit_fmt = workbook.add_format(
        {'italic': True, 'bg_color': '#F4B084', 'align': 'left', 'bold': True})
    title_fmt = workbook.add_format(
        {'bg_color': '#0070C0', 'font_color': '#FFFFFF', 'align': 'center', 'valign': 'vcenter', 'bold': True})
    capacity_fmt = workbook.add_format(
        {'bg_color': '#00B0F0', 'align': 'center', 'bold': True})
    group_info_fmt = workbook.add_format(
        {'bg_color': '#FFC000', 'align': 'center'})
    route_descr_top_fmt = workbook.add_format(
        {'bg_color': '#FFFB00', 'align': 'left', 'bold': True, 'valign': 'vcenter', 'top': 1})
    route_descr_bot_fmt = workbook.add_format(
        {'bg_color': '#FFFB00', 'align': 'left', 'bold': True, 'valign': 'vcenter', 'bottom': 1})
    tot_stay_title_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'align': 'right', 'bold': True})
    tot_stay_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'align': 'center', 'bold': True, 'top': 1})
    tot_stay_bot_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'align': 'center', 'bold': True, 'bottom': 1})
    remaining_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'font_color': 'red', 'align': 'center', 'bold': True})
    entry_fmt = workbook.add_format({'bg_color': '#00B050', 'align': 'center'})
    stay_fmt = workbook.add_format({'bg_color': '#92D050', 'align': 'center'})
    empty_hotel_fmt = workbook.add_format(
        {'bg_color': '#D9D9D9', 'align': 'right'})
    hint_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#CCC0DA', 'align': 'right'})
    hint_perc_fmt = workbook.add_format(
        {'num_format': 10, 'bg_color': '#CCC0DA', 'align': 'right'})
    hint_bold_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#CCC0DA', 'bold': True, 'align': 'right'})
    hint_bold_percent_fmt = workbook.add_format(
        {'num_format': '0.00%', 'bg_color': '#CCC0DA', 'bold': True, 'align': 'right'})
    hint_err_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#CCC0DA', 'bold': True, 'font_color': '#FF2600', 'align': 'right'})
    hint_err_detail_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#CCC0DA', 'font_color': '#C00000', 'align': 'right'})
    tot_lost_seat_fmt = workbook.add_format(
        {'bg_color': '#D7A284', 'font_color': '#C00000', 'italic': True})

    # End formatting definition
    # Making white background
    [ws.write(i, j, "", blank_fmt) for i in range(800) for j in range(70)]
    ws.set_zoom(100)
    ws.freeze_panes(8, 3)
    ws.set_column(3, 2+day_count, 4.83)
    ws.write(0, 0, "Objective Value:", obj_fmt)
    ws.write(0, 1, "", obj_fmt)
    ws.write(0, 2, objective_value, obj_fmt)
    ws.write(0, 3, np.sum(XRes), obj3_fmt)
    ws.write(0, 4, objective_value-prob.tot_pass-np.sum(XRes), obj2_fmt)
    for d in range(day_count):
        ws.write(0, 5+d, "", blank_fmt)

    # Total Entry Row
    # Calculation
    total_med_f = np.zeros(day_count)
    total_med_pass = np.zeros(day_count)
    total_jed_f = np.zeros(day_count)
    total_jed_pass = np.zeros(day_count)
    total_f = np.zeros(day_count)
    total_pass = np.zeros(day_count)

    for d in range(day_count):
        for f in range(flight_count):
            for r in range(route_count):
                if (r < len(prob.station_passengers)):
                    total_med_f[d] += X[f, d, r]
                    total_med_pass[d] += X[f, d, r] * \
                        (flight_capacity[f] - reserve[d, r])-XRes[f, d, r]
                else:
                    total_jed_f[d] += X[f, d, r]
                    total_jed_pass[d] += X[f, d, r] * \
                        (flight_capacity[f] - reserve[d, r])-XRes[f, d, r]
                total_f[d] += X[f, d, r]
                total_pass[d] += X[f, d, r]*(flight_capacity[f] - reserve[d, r])-XRes[f, d, r]

    print("X")
    print(X.tolist())
    print(XRes.tolist())
    print("flight_capacity")
    print(flight_capacity)
    print("reserve")
    print(reserve.tolist())

    print("total_med_pass")
    print(total_med_pass.tolist())
    print("total_jed_pass")
    print(total_jed_pass.tolist())
    # Fill med_last_downhaul_start_day
    global med_last_backhaul_start_day
    med_last_backhaul_start_day = int(total_jed_pass.nonzero()[0][0] + 1)
    # End Calculation

    ws.write(1, 0, 'Total Jeddah Flights', tot_entry_title_fmt)
    ws.write(1, 1, '', tot_entry_fmt)
    ws.write(1, 2, '', tot_entry_fmt)
    for i in range(day_count):
        ws.write(1, 3+i, total_med_f[i], tot_entry_fmt)

    ws.write(2, 0, 'Total Medinah Flights', tot_entry_title_fmt)
    ws.write(2, 1, '', tot_entry_fmt)
    ws.write(2, 2, '', tot_entry_fmt)
    for i in range(day_count):
        ws.write(2, 3+i, total_jed_f[i], tot_entry_fmt)

    ws.write(3, 0, 'Total Flights', tot_entry_diff_fmt)
    ws.write(3, 1, '', tot_entry_diff_fmt)
    ws.write(3, 2, '', tot_entry_diff_fmt)
    for i in range(day_count):
        ws.write(3, 3+i, total_f[i], tot_entry_diff_fmt)

    ws.write(4, 0, 'Total Jeddah Passengers', tot_entry_title_fmt)
    ws.write(4, 1, '', tot_entry_fmt)
    ws.write(4, 2, '', tot_entry_fmt)
    for i in range(day_count):
        ws.write(4, 3+i, total_med_pass[i], tot_entry_fmt)

    ws.write(5, 0, 'Total Medinah Passengers', tot_entry_title_fmt)
    ws.write(5, 1, '', tot_entry_fmt)
    ws.write(5, 2, '', tot_entry_fmt)
    for i in range(day_count):
        ws.write(5, 3+i, total_jed_pass[i], tot_entry_fmt)

    ws.write(6, 0, 'Total Passengers', tot_entry_diff_tit_fmt)
    ws.write(6, 1, '', tot_entry_diff_fmt)
    ws.write(6, 2, '', tot_entry_diff_fmt)
    for i in range(day_count):
        ws.write(6, 3+i, total_pass[i], tot_entry_diff_fmt)

    # Data Cells
    row = 7
    col = 3+day_count  # for entering hints
    for f in range(flight_count):
        # Header Row - Number of Days
        ws.write(row+f*(3+route_count), 0, "Flight / Day", title_fmt)
        ws.write(row+f*(3+route_count), 1, "Route", group_info_fmt)
        ws.write(row+f*(3+route_count), 2, "", group_info_fmt)
        for i in range(day_count):
            ws.write(row+f*(3+route_count), 3+i, i+1, title_fmt)

        # Header Row - Total Stay & Remaining Cap
        ws.write(row+f*(3+route_count)+1, 0, "Capacity", capacity_fmt)
        ws.write(row+f*(3+route_count)+2, 0, flight_capacity[f], capacity_fmt)

        ws.write(row+f*(3+route_count)+1, 2,
                 "Total Flight", tot_stay_title_fmt)
        ws.write(row+f*(3+route_count)+1, 1, "", tot_stay_fmt)
        ws.write(row+f*(3+route_count)+2, 2,
                 "Total Passenger", tot_stay_title_fmt)
        ws.write(row+f*(3+route_count)+2, 1, "", tot_stay_fmt)

        # Calculating Total Stay for Hotel h
        total_flight = np.zeros(day_count)
        total_flight_pass = np.zeros(day_count)

        for d in range(day_count):
            for r in range(route_count):
                total_flight[d] += X[f, d, r]
                total_flight_pass[d] += X[f, d, r] * \
                    (flight_capacity[f] - reserve[d, r])-XRes[f, d, r]
        # End Calculation
        for d in range(day_count):
            ws.write(row+f*(3+route_count)+1, 3+d,
                     total_flight[d], tot_stay_fmt)
            if total_flight[d] > 0:
                ws.write(row+f*(3+route_count)+2, 3+d,
                         total_flight_pass[d], remaining_fmt)
            # Formating empty hotel days
            else:
                ws.write(row+f*(3+route_count)+2, 3+d, "", remaining_fmt)
                for i in range(route_count):
                    ws.write(row+f*(3+route_count)+3+i,
                             3+d, "", empty_hotel_fmt)

        # Merge Hotel Numbers
        ws.merge_range(row+f*(3+route_count)+3, 0, row+f *
                       (3+route_count)+3+route_count-1, 0, f+1, title_fmt)
        for r in range(route_count):
            ws.write(row+f*(3+route_count)+3+r, 1, r+1, group_info_fmt)
            ws.write(row+f*(3+route_count)+3+r, 2,
                     route_descr[r], group_info_fmt)

            for d in range(day_count):
                if X[f, d, r] < 1:
                    continue
                # if d < day_count-stay_duration+1 and X[f,d,r] >= 1:
                if X[f, d, r] >= 1:
                    ws.write(row+f*(3+route_count)+3+r, 3+d,
                             X[f, d, r], entry_fmt)
                else:
                    ws.write(row+f*(3+route_count)+3+r, 3+d,
                             X[f, d, r], stay_fmt)
        row += 1

    ws = workbook.add_worksheet('Route Based')

    # Making white background
    [ws.write(i, j, "", blank_fmt) for i in range(800) for j in range(70)]
    ws.set_zoom(95)
    ws.freeze_panes(10, 3)
    ws.set_column(3, 2+day_count, 4.83)
    ws.write(0, 0, "Objective Value:", obj_fmt)
    ws.write(0, 1, "", obj_fmt)
    ws.write(0, 2, objective_value, obj_fmt)
    ws.write(0, 3, np.sum(XRes), obj3_fmt)
    ws.write(0, 4, objective_value-prob.tot_pass-np.sum(XRes), obj2_fmt)
    for d in range(day_count):
        ws.write(0, 5+d, "", blank_fmt)

    # Total Entry Row
    # Calculation
    total_med_f = np.zeros(day_count)
    total_med_pass = np.zeros(day_count)
    total_jed_f = np.zeros(day_count)
    total_jed_pass = np.zeros(day_count)
    total_f = np.zeros(day_count)
    total_pass = np.zeros(day_count)

    for d in range(day_count):
        for f in range(flight_count):
            for r in range(route_count):
                if (r < len(prob.station_passengers)):
                    total_med_f[d] += X[f, d, r]
                    total_med_pass[d] += X[f, d, r] * \
                        (flight_capacity[f] - reserve[d, r])-XRes[f, d, r]
                else:
                    total_jed_f[d] += X[f, d, r]
                    total_jed_pass[d] += X[f, d, r] * \
                        (flight_capacity[f] - reserve[d, r])-XRes[f, d, r]
                total_f[d] += X[f, d, r]
                total_pass[d] += X[f, d, r]*(flight_capacity[f] - reserve[d, r])-XRes[f, d, r]
    # End Calculation

    ws.write(1, 0, 'Total Jeddah Flights', tot_entry_title_fmt)
    ws.write(1, 1, '', tot_entry_fmt)
    ws.write(1, 2, '', tot_entry_fmt)
    for i in range(day_count):
        ws.write(1, 3+i, total_med_f[i], tot_entry_fmt)

    ws.write(2, 0, 'Total Medinah Flights', tot_entry_title_fmt)
    ws.write(2, 1, '', tot_entry_fmt)
    ws.write(2, 2, '', tot_entry_fmt)
    for i in range(day_count):
        ws.write(2, 3+i, total_jed_f[i], tot_entry_fmt)

    ws.write(3, 0, 'Total Flights', tot_entry_diff_tit_fmt)
    ws.write(3, 1, '', tot_entry_diff_fmt)
    ws.write(3, 2, '', tot_entry_diff_fmt)
    for i in range(day_count):
        ws.write(3, 3+i, total_f[i], tot_entry_diff_fmt)

    ws.write(4, 0, 'Total Jeddah Passengers', tot_entry_title_fmt)
    ws.write(4, 1, '', tot_entry_fmt)
    ws.write(4, 2, '', tot_entry_fmt)
    for i in range(day_count):
        ws.write(4, 3+i, total_med_pass[i], tot_entry_fmt)

    ws.write(4, col, sum(total_med_pass), hint_fmt)
    ws.write(4, col+1, sum(total_med_pass) /
             (sum(total_med_pass)+sum(total_jed_pass)), hint_perc_fmt)

    ws.write(6, 0, 'Total Medinah Passengers', tot_entry_title_fmt)
    ws.write(6, 1, '', tot_entry_fmt)
    ws.write(6, 2, '', tot_entry_fmt)
    for i in range(day_count):
        ws.write(6, 3+i, total_jed_pass[i], tot_entry_fmt)

    ws.write(6, col, sum(total_jed_pass), hint_fmt)
    ws.write(6, col+1, sum(total_jed_pass) /
             (sum(total_med_pass)+sum(total_jed_pass)), hint_perc_fmt)

    ws.write(8, 0, 'Total Passengers', tot_entry_diff_tit_fmt)
    ws.write(8, 1, '', tot_entry_diff_fmt)
    ws.write(8, 2, '', tot_entry_diff_fmt)
    for i in range(day_count):
        ws.write(8, 3+i, total_pass[i], tot_entry_diff_fmt)

    # Header Row - Number of Days
        ws.write(9, 0, "Route / Day", title_fmt)
        ws.write(9, 1, "Flight", group_info_fmt)
        ws.write(9, 2, "", group_info_fmt)
        for i in range(day_count):
            ws.write(9, 3+i, i+1, title_fmt)

        ws.write(9, col, "Total", hint_bold_fmt)
        ws.write(9, col+1, "Pass/Lost", hint_bold_fmt)
        ws.write(9, col+2, "Lost Seat", hint_bold_fmt)
        ws.write(9, col+3, "Duration", hint_bold_fmt)

    # Data Cells
    row = 10

    last_total_pass_in_route = 0
    for s in range(len(prob.station_passengers)):
        for c in range(2):
            r = 2*s+c
            i = s+c*len(prob.station_passengers)
            ws.merge_range(row+r*(1+flight_count), 0,
                           row+r*(1+flight_count)+flight_count, 0, i+1, title_fmt)

            ws.write(row+r*(1+flight_count), 1,
                     "Total", tot_stay_title_fmt)

            if c == 0:
                ws.merge_range(row+r*(1+flight_count), 2,
                               row+r*(1+flight_count)+flight_count, 2, route_descr[i], route_descr_top_fmt)
            else:
                ws.merge_range(row+r*(1+flight_count), 2,
                               row+r*(1+flight_count)+flight_count, 2, route_descr[i], route_descr_bot_fmt)

            ws.conditional_format(row+r*(1+flight_count), 3,
                                  row+route_count*(1+flight_count), 2+day_count, {'type': '3_color_scale'})

            total_flights_in_route = np.zeros(flight_count)
            for d in range(day_count):
                tot = 0
                for f in range(flight_count):
                    tot += X[f, d, i]
                    total_flights_in_route[f] += X[f, d, i]

                if c == 0:
                    if tot > 0:
                        ws.write(row+r*(1+flight_count),
                                 3+d, tot, tot_stay_fmt)
                    else:
                        ws.write(row+r*(1+flight_count), 3+d, "", tot_stay_fmt)
                else:
                    if tot > 0:
                        ws.write(row+r*(1+flight_count),
                                 3+d, tot, tot_stay_bot_fmt)
                    else:
                        ws.write(row+r*(1+flight_count),
                                 3+d, "", tot_stay_bot_fmt)

            total_pass_in_route = 0
            for f in range(flight_count):
                total_pass_in_route += total_flights_in_route[f] * \
                    flight_capacity[f] - np.sum(XRes[f, :, i])

            ws.write(row+r*(1+flight_count), col,
                     total_pass_in_route, hint_bold_fmt)

            ws.write(row+r*(1+flight_count), col+3,
                     max_duration[i], hint_bold_fmt)

            if c == 0:
                ws.write(row+r*(1+flight_count), col+1,
                         prob.station_passengers[s], hint_bold_fmt)
            else:
                extra_seats = total_pass_in_route + \
                    last_total_pass_in_route - prob.station_passengers[s]
                ws.write(row+r*(1+flight_count), col +
                         1, extra_seats, hint_err_fmt)
                if extra_seats > 0:
                    extra_seats_in_routes[s] = extra_seats

            for f in range(flight_count):
                ws.set_row(row+r*(1+flight_count)+f +
                           1, None, None, {'level': 1})
                ws.write(row+r*(1+flight_count)+f+1, 1, f+1, group_info_fmt)
                for d in range(day_count):
                    ws.write(row+r*(1+flight_count)+f+1, 3+d,
                             X[f, d, i], entry_fmt)
                    if XRes[f, d, i] > 0:
                        ws.write_comment(xl_rowcol_to_cell(row+r*(1+flight_count)+f+1, 3+d),
                                         "Free Empty Seats: \n%i" % (XRes[f, d, i]), {'width': 113, 'height': 38, 'font_name': 'Calibri', 'font_size': 10})

                ws.write(row+r*(1+flight_count)+f+1, col,
                         total_flights_in_route[f]*flight_capacity[f], hint_fmt)

            last_total_pass_in_route = total_pass_in_route

    # NEW: To remove staff reserve from extra seats
    extra_seats_in_routes[0] = 0
    # Lost Seats

    for s in range(len(prob.station_passengers)):
        for c in range(2):
            i = s+c*len(prob.station_passengers)
            for d in range(day_count):
                for f in range(flight_count):
                    if extra_seats_in_routes[s] > 0 and X[f, d, i] > 0:
                        if c == 0:
                            if extra_seats_in_routes_days_med[s].count(d) == 0:
                                extra_seats_in_routes_days_med[s].append(d)
                            extra_seats_in_routes_flight_capacity_weight_med[
                                s] += (flight_capacity[f] - reserve[d, i])*X[f, d, i]-XRes[f, d, i]
                        else:
                            if extra_seats_in_routes_days_mec[s].count(d) == 0:
                                extra_seats_in_routes_days_mec[s].append(d)
                            extra_seats_in_routes_flight_capacity_weight_mec[
                                s] += (flight_capacity[f] - reserve[d, i])*X[f, d, i]-XRes[f, d, i]

    for s in range(len(prob.station_passengers)):
        if extra_seats_in_routes[s] > 0:
            i = s
            # if c == 0:
            for d in extra_seats_in_routes_days_med[s]:
                w = 0
                for f in range(flight_count):
                    if X[f, d, i] > 0:
                        w += X[f, d, i]*(flight_capacity[f] - reserve[d, i])-XRes[f, d, i]
                        # Flight_Route Lost Seats
                        lost_seats_fr[f, i] += extra_seats_in_routes[s] / \
                            (extra_seats_in_routes_flight_capacity_weight_med[s] +
                             extra_seats_in_routes_flight_capacity_weight_mec[s]) * (X[f, d, i]*(flight_capacity[f] - reserve[d, i])-XRes[f, d, i])

                lost_seats_med[d] += extra_seats_in_routes[s] / \
                    (extra_seats_in_routes_flight_capacity_weight_med[s] +
                     extra_seats_in_routes_flight_capacity_weight_mec[s]) * w

            i = s+len(prob.station_passengers)
            for d in extra_seats_in_routes_days_mec[s]:
                w = 0
                for f in range(flight_count):
                    if X[f, d, i] > 0:
                        w += X[f, d, i]*(flight_capacity[f] - reserve[d, i])-XRes[f, d, i]
                        # Flight_Route Lost Seats
                        lost_seats_fr[f, i] += extra_seats_in_routes[s] / \
                            (extra_seats_in_routes_flight_capacity_weight_med[s] +
                             extra_seats_in_routes_flight_capacity_weight_mec[s]) * (X[f, d, i]*(flight_capacity[f] - reserve[d, i])-XRes[f, d, i])

                lost_seats_mec[d] += extra_seats_in_routes[s] / \
                    (extra_seats_in_routes_flight_capacity_weight_med[s] +
                     extra_seats_in_routes_flight_capacity_weight_mec[s]) * w

    ws.write(5, 0, 'Total Med-First Lost Seats', tot_lost_seat_fmt)
    ws.write(5, 1, '', tot_lost_seat_fmt)
    ws.write(5, 2, '', tot_lost_seat_fmt)

    for d in range(day_count):
        ws.write(5, 3+d, int(lost_seats_med[d]), tot_lost_seat_fmt)

    ws.write(7, 0, 'Total Med-Last Lost Seats', tot_lost_seat_fmt)
    ws.write(7, 1, '', tot_lost_seat_fmt)
    ws.write(7, 2, '', tot_lost_seat_fmt)
    for d in range(day_count):
        ws.write(7, 3+d, int(lost_seats_mec[d]), tot_lost_seat_fmt)

    # Flight_Route Lost Seats write to excel
    lost_by_flight = np.sum(lost_seats_fr, axis=1).astype(int)
    ws.write(0, col+3, "Flight", hint_bold_fmt)
    ws.write(0, col+4, "Total", hint_bold_fmt)
    ws.write(0, col+5, "Percent", hint_bold_fmt)
    ws.write(0, col+6, "Lost Seats", hint_bold_fmt)
    for f in range(flight_count):
        ws.write(f+1, col+3, f+1, hint_bold_fmt)
        ws.write(f+1, col+6, lost_by_flight[f], hint_bold_fmt)
        # Total
        tot_f = 0
        for d in range(day_count):
            for r in range(route_count):
                tot_f += X[f, d, r]*(flight_capacity[f] - reserve[d, r])-XRes[f, d, r]
        ws.write(f+1, col+4, tot_f, hint_bold_fmt)
        ws.write(f+1, col+5, tot_f/(objective_value -
                 np.sum(XRes)), hint_bold_percent_fmt)
    ws.write(flight_count+1, col+3, "Total", hint_err_fmt)
    ws.write(flight_count+1, col+4, objective_value -
             np.sum(XRes), hint_err_fmt)
    ws.write(flight_count+1, col+5, "100%", hint_err_fmt)
    ws.write(flight_count+1, col+6, objective_value -
             prob.tot_pass - np.sum(XRes), hint_err_fmt)

    lost_by_route = np.sum(lost_seats_fr, axis=0).astype(int)
    for s in range(len(prob.station_passengers)):
        ws.write(row, col+2, lost_by_route[s], hint_err_detail_fmt)
        row += 1+flight_count
        ws.write(
            row, col+2, lost_by_route[s+len(prob.station_passengers)], hint_err_detail_fmt)
        row += 1+flight_count

    workbook.close()


# def main():
#     prob = Problem(
#         downhaul_tot_days=30,
#         down_end_to_back_start_days=11,
#         tot_days=23,
#         days1=19,
#         days2_start=6,
#         tot_pass=87552,
#         both_stations=[0, 1, 2],
#         station_passengers=[25108, 11184, 8621, 4194, 4660, 3728, 3029, 699,
#                             699, 4427, 1165, 1631, 932, 2097, 2796, 1165, 932, 2563, 4194, 3728],
#         flight_capacity=[[233, 250], [274, 290], [300, 430]],
#         flight_capacity_toler=[5, 5, 5, 6, 6, 9],
#         flight_cap_downhaul=[233, 250, 274, 290, 300, 430],
#         airline_daily_max=[11, 5, 9],
#         airline_total_share=[0.42, 0.11, 0.47],
#         airline_total_share_tolerance=0.02,
#         total_daily_slot_limit=[4570, 4860],
#         daily_pass_limit_min=1830,
#         daily_pass_limit_max=2220,
#         daily_slot_cap_small_cities=3,
#         daily_slot_cap_mid_cities=4)

#     constr = Constraints(
#         downhaul_sol="downhaul.sol",
#         unavailable_flights_in_routes={
#             4: [1, 21, 6, 26, 8, 28, 9, 29, 10, 20, 11, 31, 13, 33, 16, 36, 17, 37, 18, 38],
#             5: list(set(list(range(40))) - set([0, 2, 20, 22]))},
#         flights_daily_cap={4: [[5]]},
#         glued_flights=[[5]],
#         small_stations_routes=list(
#             set(list(range(40))) - set([0, 1, 2, 20, 21, 22])),
#         mid_stations_routes=[1, 21, 2, 22],
#     )

#     try:
#         # create_flight_model(prob, constr)
#         None
#     finally:
#         if opt_model.status == 11:
#             print("Model interrupted.")
#         # opt_model.write("backhaul.sol")
#         visualize(prob)


# if __name__ == "__main__":
#     main()
