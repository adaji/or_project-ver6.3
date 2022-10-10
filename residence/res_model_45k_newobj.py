from re import X
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import xlsxwriter as xw

# This is the 7th model for Haj Project - Completely Rewritten in Gurobi fashion
# This model uses X(hotel, days, group CLASSES) with size (18,31,16) as input variables.
# Difference with version 7: Parametric - No carevan class, just sizes matter - input daily flight cap

opt_model = gp.Model(name="Residenece MIP Model")

x_vars = []
y_vars = []

runtime = 0
objbst = 0
objbnd = 0
solcnt = 0
gap = 0.0


class Problem:
    def __init__(self,
                 med_first=1,
                 daily_flight_med=[],
                 station_passengers=[],
                 stay_duration=6,
                 hotel_price=[0.5, 0.6, 1, 0.6, 0.5, 0.6, 0.5, 0.8,
                              0.8, 0.5, 0.6, 0.6, 0.6, 0.6, 0.8, 0.6, 0.8, 0.6],
                 hotel_capacity=[819, 819, 1073, 514, 457, 667, 791, 486,
                                 1520, 1085, 1000, 892, 486, 463, 542, 597, 667, 429],
                 hotel_availability=[[], [], [], [], [], [], [],
                                     [], [], [], [], [], [], [], [], [], [], []],
                 ) -> None:
        self.med_first = med_first
        self.daily_flight_med = daily_flight_med
        self.station_passengers = station_passengers
        self.stay_duration = stay_duration
        self.hotel_price = hotel_price
        self.hotel_capacity = hotel_capacity
        self.hotel_availability = hotel_availability


class Constraints:
    def __init__(self,
                 min_hotel_fill=[0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 daily_hotel_entry_cap=[0.7, 0.7, 1, 1, 1, 0.8, 0.7, 1,
                                        0.7, 0.7, 0.7, 0.7, 1, 1, 0.8, 0.8, 0.8, 1],
                 paying_early_days=8,
                 paying_last_days=7,
                 carevan_size_mult=45,
                 carevan_size_res=3,
                 remove_daily_mult_above_1=2
                 ) -> None:

        self.min_hotel_fill = min_hotel_fill
        self.daily_hotel_entry_cap = daily_hotel_entry_cap
        self.paying_early_days = paying_early_days
        self.paying_last_days = paying_last_days
        self.carevan_size_mult = carevan_size_mult
        self.carevan_size_res = carevan_size_res
        self.remove_daily_mult_above_1 = remove_daily_mult_above_1


def create_res_model(prob=Problem(), constr=Constraints()):
    entry_days = len(prob.daily_flight_med)
    tot_days = entry_days + prob.stay_duration - 1  # D:31 / B:23

    tot_passengers = sum(prob.daily_flight_med)  # - sum(daily_flight_toler)

    num_hotels = len(prob.hotel_price)

    all_hotels = range(num_hotels)
    all_days = range(entry_days+prob.stay_duration-1)  # Down 31 back 23

    # entry[(h,d)]: Number of people that enters hotel 'h' on day 'd'.
    x = {(h, d): opt_model.addVar(vtype=GRB.INTEGER, name='X(%i_%i)' % (h, d))
         for h in all_hotels for d in range(entry_days)}

    # Auxilary Var - Stay
    stay = {(h, d): opt_model.addVar(vtype=GRB.BINARY, name='Y(%i_%i)' % (h, d))
            for h in all_hotels for d in all_days}

    # Auxiliary Variables
    opt_model.addConstrs(1000000 * stay[h, d] >= gp.quicksum([x[h, i] for i in range(max(d-prob.stay_duration+1, 0), min(d+1, entry_days))])
                         for d in all_days for h in all_hotels)
    opt_model.addConstrs(stay[h, d] <= gp.quicksum([x[h, i] for i in range(max(d-prob.stay_duration+1, 0), min(d+1, entry_days))])
                         for d in all_days for h in all_hotels)

    # All passengers should enter
    opt_model.addConstr(gp.quicksum(x[h, d] for h in all_hotels for d in range(
        entry_days)) == 17285)  # tot_passengers)

    # Remove hotels in days that are not available
    rem = []
    for i in prob.hotel_availability:
        s = set(range(entry_days))
        for j in i:
            s -= set(range(j[0], j[1]+1))
        s = list(s) if len(s) < entry_days else []
        rem.append(s)

    opt_model.addConstrs(x[h, d] == 0 for h in all_hotels for d in rem[h])

    # x which is the daily entry in hotels, should be 45x +-3
    xmultiple = {(h, d): opt_model.addVar(vtype=GRB.INTEGER, lb=0, name='XMULT(%i_%i)' % (h, d))
                 for h in all_hotels for d in range(entry_days)}

    xmult_aux = {(h, d): opt_model.addVar(vtype=GRB.BINARY, name='AUXMULT(%i_%i)' % (h, d))
                 for h in all_hotels for d in range(entry_days)}

    xresidual = {(h, d): opt_model.addVar(vtype=GRB.INTEGER, name='XRES(%i_%i)' % (h, d))
                 for h in all_hotels for d in range(entry_days)}

    opt_model.addConstrs(x[h, d] == constr.carevan_size_mult * xmultiple[h, d] + xresidual[h, d]
                         for h in all_hotels for d in range(entry_days))

    # M should not be 1. group size must be larger than 90 (m>=2) or m=0
    opt_model.addConstrs(xmultiple[h, d] >= (1+constr.remove_daily_mult_above_1) - 100000000 * xmult_aux[h, d]
                         for h in all_hotels for d in range(entry_days))
    opt_model.addConstrs(xmultiple[h, d] <= 100000000 * (1 - xmult_aux[h, d])
                         for h in all_hotels for d in range(entry_days))

    # Ensure if multiple is 0, residual is zero too. (Problem with some x only equal to residual like 4)
    # opt_model.addConstrs(x[h, d] <= 100000 * xmultiple[h, d]
    #                      for h in all_hotels for d in range(entry_days))
    # NEW way: r<= 3m & r>=-3m Advantage: each 45x group can have +- 3 size
    opt_model.addConstrs(xresidual[h, d] <= constr.carevan_size_res * xmultiple[h, d]
                         for h in all_hotels for d in range(entry_days))
    opt_model.addConstrs(xresidual[h, d] >= -constr.carevan_size_res * xmultiple[h, d]
                         for h in all_hotels for d in range(entry_days))

    # Hotel minimum stay assurance
    opt_model.addConstrs(gp.quicksum(x[h, d] for d in range(entry_days)) >= constr.min_hotel_fill[h]
                         for h in all_hotels)

    # Restrict hotel capacity
    opt_model.addConstrs(
        gp.quicksum(x[h, i] for i in range(max(d-prob.stay_duration+1, 0),
                    min(d+1, entry_days))) <= prob.hotel_capacity[h]
        for d in all_days for h in all_hotels)

    # Limit daily entry in each hotel
    opt_model.addConstrs(x[h, d] <= prob.hotel_capacity[h] * constr.daily_hotel_entry_cap[h]
                         for h in all_hotels for d in range(entry_days))

    # Daily Entrance Limit (Flight limits): *********IMPORTANT: CAP OR FLOOR?
    # opt_model.addConstrs(gp.quicksum(x[h, d] for h in all_hotels) >= prob.daily_flight_med[d] - 300  # - daily_flight_toler[d]  # 46
    #                      for d in range(entry_days))

    opt_model.addConstrs(gp.quicksum(x[h, d] for h in all_hotels) <= prob.daily_flight_med[d]  # - daily_flight_toler[d]  # 46
                         for d in range(entry_days))

    # Limit empty seat usage
    # NEW METHOD FOR LIMITING EMPTY SEAT USED
    # pos_beds = {(d): opt_model.addVar(vtype=GRB.INTEGER, lb=0, name='POSBED(%i)' % (d))
    #             for d in range(entry_days)}
    # neg_beds = {(d): opt_model.addVar(vtype=GRB.INTEGER, lb=0, name='NEGBED(%i)' % (d))
    #             for d in range(entry_days)}
    # opt_model.addConstrs(
    #     pos_beds[d]-neg_beds[d] == gp.quicksum([x[h, d]
    #                                            for h in all_hotels]) - daily_flight_cap[d]
    #     for d in range(entry_days))

    # opt_model.addConstrs(neg_beds[d] <= daily_flight_toler[d]
    #                      for d in range(entry_days))
    # Main Cost Objective
    obj1 = gp.quicksum(stay[h, d]*prob.hotel_capacity[h]*prob.hotel_price[h]
                       for d in all_days for h in all_hotels)

    # Stay 2 - for considering cost of hotel if early allowed days or last allowed empty days are filled
    stay2 = {(h, d): opt_model.addVar(vtype=GRB.BINARY, name='Y2(%i_%i)' % (h, d))
             for h in all_hotels for d in all_days}
    opt_model.addConstrs(stay2[h, d] == gp.any_([stay[h, i] for i in range(constr.paying_early_days)])
                         for d in range(constr.paying_early_days) for h in all_hotels)

    opt_model.addConstrs(stay2[h, d] == 0 for d in range(
        constr.paying_early_days, tot_days-constr.paying_last_days) for h in all_hotels)

    opt_model.addConstrs(stay2[h, d] == gp.any_([stay[h, i] for i in range(tot_days-constr.paying_last_days, tot_days)])
                         for d in range(tot_days-constr.paying_last_days, tot_days) for h in all_hotels)
    opt_model.addConstrs(1000000 * stay2[h, d] >= gp.quicksum([x[h, i] for i in range(tot_days -
                                                                                      prob.stay_duration+1-constr.paying_last_days, min(d, entry_days))])
                         for d in range(tot_days-constr.paying_last_days, tot_days) for h in all_hotels)
    opt_model.addConstrs(stay2[h, d] <= gp.quicksum([x[h, i] for i in range(tot_days -
                                                                            prob.stay_duration+1-constr.paying_last_days, min(d, entry_days))])
                         for d in range(tot_days-constr.paying_last_days, tot_days) for h in all_hotels)

    obj2 = gp.quicksum(stay2[h, d]*prob.hotel_capacity[h]*prob.hotel_price[h]
                       for d in all_days for h in all_hotels)

    # New objective - goal: remove gap between stays in each hotel
    # 3 Auxiliary variables
    # w1 = {(h, d): opt_model.addVar(vtype=GRB.BINARY, name='w1(%i_%i)' % (h, d))
    #       for h in all_hotels for d in all_days}
    # w2 = {(h, d): opt_model.addVar(vtype=GRB.BINARY, name='w2(%i_%i)' % (h, d))
    #       for h in all_hotels for d in all_days}
    # z = {(h, d): opt_model.addVar(vtype=GRB.BINARY, name='z(%i_%i)' % (h, d))
    #      for h in all_hotels for d in all_days}

    # # w1 constraints - if earlier days have stay, 1, otherwise 0
    # opt_model.addConstrs(w1[h, d] == gp.any_(stay[h, i] for i in range(d))
    #                      for h in all_hotels for d in all_days)

    # # w2 constraints - if later days have stay, 1, otherwise 0
    # opt_model.addConstrs(w2[h, d] == gp.any_(stay[h, i] for i in range(min(d+1, tot_days), tot_days))
    #                      for h in all_hotels for d in all_days)

    # # z constraints - if w1 and w2 are true, 1, otherwise 0
    # opt_model.addConstrs((stay[h, d] == 0) >> (z[h, d] >= w1[h, d] + w2[h, d] - 1)
    #                      for h in all_hotels for d in all_days)

    # obj3 = gp.quicksum(z[h, d]*prob.hotel_capacity[h]*prob.hotel_price[h]
    #                    for h in all_hotels for d in all_days)

    # NEW OBJECTIVE: GOAL is to limit carevan sizes to be near a target number (% of tot pass)

    opt_model.ModelSense = GRB.MINIMIZE
    opt_model.setObjective(obj1+obj2)

    if prob.med_first == 1:
        opt_model.write("./Output/Eskan/res_model_med_first.lp")
    else:
        opt_model.write("./Output/Eskan/res_model_med_last.lp")

    # Model Parameters
    # opt_model.Params.timelimit = 2000  # 1400
    opt_model.Params.integralityfocus = 1
    # opt_model.Params.mipfocus = 1
    # opt_model.Params.seed = 123
    # opt_model.Params.heuristics = 0.97

    def getSol(model, where):
        if where == GRB.Callback.MIPSOL:
            global x_vars
            global y_vars
            global solcnt

            x_vars = model.cbGetSolution(x)
            y_vars = model.cbGetSolution(stay)
            solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)

        elif where == GRB.Callback.MIP:
            global runtime
            global objbst
            global objbnd
            global gap

            runtime = model.cbGet(GRB.Callback.RUNTIME)
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            gap = abs(objbst-objbnd)/objbst

    opt_model.optimize(getSol)


def visualize(prob, sol_x, sol_y, response):

    # Computing daily flight cap from flight solution
    # X0 = np.array(prob.flight_sol)
    # X0Res = np.array(prob.flight_sol_res)
    # daily_flight_cap = []

    # routes = []
    # if prob.med_first == 1:
    #     routes = range(len(prob.station_passengers))
    # else:
    #     routes = range(len(prob.station_passengers), len(prob.station_passengers) * 2)

    # for d in range(X0.shape[1]):
    #     daily_cap = 0
    #     for f in range(len(prob.flight_capacity)):
    #         for r in routes:
    #             daily_cap += X0[f, d, r] * prob.flight_capacity[f] - X0Res[f, d, r]
    #     if daily_cap == 0:
    #         continue
    #     daily_flight_cap.append(daily_cap)

    entry_days = len(prob.daily_flight_med)
    tot_days = entry_days + prob.stay_duration - 1  # D:31 / B:23

    hotel_count = len(prob.hotel_capacity)
    # Get Data from gloal var instead of .sol file
    # with open(prob.solution_path) as f:
    #     content = f.read()
    #     entry = re.findall(r'X\((\d+)_(\d+)\) (\d+)', content)
    #     entry_list = np.zeros(
    #         (hotel_count, tot_days - prob.stay_duration+1), dtype=int)
    #     for x in entry:
    #         # Remove -1 if using optimal solution - -1 just for X0
    #         entry_list[int(x[0]), int(x[1])] = int(x[2])
    #     # **************************
    #     stay = re.findall(r'Y\((\d+)_(\d+)\) 1', content)
    #     hotel_filled = np.zeros((hotel_count, tot_days), dtype=int)
    #     for y in stay:
    #         hotel_filled[int(y[0]), int(y[1])] = 1
    entry_list = sol_x
    hotel_filled = sol_y

    # Creating data for schedule table
    # Create stay data
    stay_list = np.zeros((hotel_count, tot_days), dtype=int)
    for h in range(hotel_count):
        for d in range(tot_days):
            for i in range(max(d-prob.stay_duration+1, 0), min(d+1, entry_days)):
                stay_list[h, d] += entry_list[h, i]
                # If in input data we don't have Y[h,d], we can calculate it by uncommenting below
                # if stay_list[h, d] > 0:
                #     hotel_filled[h, d] = 1

    # Calculating objective function value for current solution
    objective_value = 0

    for h in range(hotel_count):
        hotel_total_filled = 0
        for d in range(tot_days):
            hotel_total_filled += hotel_filled[h, d]
        objective_value += hotel_total_filled * prob.hotel_capacity[h] * prob.hotel_price[h]

    print("Objective value for current solution: %i" % (objective_value))

    # Creating spreadsheet
    workbook = xw.Workbook(response, {'in_memory': True})
    ws = workbook.add_worksheet('Hotels')  # ws means worksheet

    # Before writing values, define formatting
    blank_fmt = workbook.add_format({'bg_color': '#FFFFFF'})
    obj_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#003296', 'font_color': '#FFFFFF', 'bold': True})
    tot_entry_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#ED7D31', 'align': 'center', 'bold': True})
    tot_entry_diff_fmt = workbook.add_format(
        {'italic': True, 'bg_color': '#F4B084', 'align': 'center', 'bold': True})
    title_fmt = workbook.add_format(
        {'bg_color': '#0070C0', 'font_color': '#FFFFFF', 'align': 'center', 'valign': 'vcenter', 'bold': True})
    capacity_fmt = workbook.add_format(
        {'bg_color': '#00B0F0', 'align': 'center', 'bold': True})
    row_title_fmt = workbook.add_format(
        {'bg_color': '#FFC000', 'align': 'center', 'num_format': '#,##0', })
    tot_stay_title_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'align': 'right', 'bold': True})
    tot_stay_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'align': 'center', 'bold': True})
    remaining_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'font_color': 'red', 'align': 'center', 'bold': True})
    entry_fmt = workbook.add_format({'bg_color': '#00B050', 'align': 'center'})
    stay_fmt = workbook.add_format({'bg_color': '#92D050', 'align': 'center'})
    empty_hotel_fmt = workbook.add_format({'bg_color': '#D9D9D9'})

    # End formatting definition
    # Making white background
    [ws.write(i, j, "", blank_fmt) for i in range(400) for j in range(50)]
    ws.set_zoom(107)
    ws.freeze_panes(6, 3)
    ws.set_column(3, 3+tot_days, 4.83)
    ws.write(0, 0, "Objective Value:", obj_fmt)
    ws.write(0, 1, "", obj_fmt)
    ws.write(0, 2, objective_value, obj_fmt)
    for d in range(tot_days):
        ws.write(0, 3+d, "", blank_fmt)

    # Total Entry Row
    # Calculation
    total_entry = np.zeros(tot_days, dtype=int)
    for d in range(entry_days):
        for h in range(hotel_count):
            total_entry[d] += entry_list[h, d]
    # End Calculation

    ws.write(1, 0, 'Total Entry', tot_entry_fmt)
    ws.write(1, 1, '', tot_entry_fmt)
    ws.write(1, 2, '', tot_entry_fmt)
    for i in range(tot_days):
        ws.write(1, 3+i, total_entry[i], tot_entry_fmt)

    ws.write(2, 0, 'Total Expected Entry', tot_entry_fmt)
    ws.write(2, 1, '', tot_entry_fmt)
    ws.write(2, 2, '', tot_entry_fmt)
    for i in range(tot_days):
        if i <= tot_days-prob.stay_duration:
            ws.write(2, 3+i, prob.daily_flight_med[i], tot_entry_fmt)
        else:
            ws.write(2, 3+i, "", tot_entry_fmt)

    ws.write(3, 0, 'Difference', tot_entry_diff_fmt)
    ws.write(3, 1, '', tot_entry_diff_fmt)
    ws.write(3, 2, '', tot_entry_diff_fmt)
    for i in range(tot_days):
        if i <= tot_days-prob.stay_duration:
            ws.write(
                3, 3+i, prob.daily_flight_med[i]-total_entry[i], tot_entry_diff_fmt)
        else:
            ws.write(3, 3+i, "", tot_entry_diff_fmt)

    # Total Vacancy
    ws.write(4, 0, 'Vacancy', row_title_fmt)
    ws.write(4, 1, '', row_title_fmt)
    ws.write(4, 2, '', row_title_fmt)
    vacancy = np.zeros(tot_days, dtype=int)
    # will be filled at the bottom of program

    # Header Row - Number of Days
    ws.write(5, 0, "Hotel \ Day", title_fmt)
    ws.write(5, 1, "", title_fmt)
    ws.write(5, 2, "", title_fmt)
    for i in range(tot_days):
        ws.write(5, 3+i, i+1, title_fmt)
    # Data Cells
    row = 5
    col = 0
    for h in range(hotel_count):
        # Header Row - Total Stay & Remaining Cap
        ws.write(row+h*4+1, 0, "Capacity", capacity_fmt)
        ws.write(row+h*4+2, 0,
                 prob.hotel_capacity[h], capacity_fmt)

        ws.write(row+h*4+1, 2, "Total Stay", tot_stay_title_fmt)
        ws.write(row+h*4+1, 1, "", tot_stay_fmt)
        ws.write(row+h*4+2, 2,
                 "Remaining Cap", tot_stay_title_fmt)
        ws.write(row+h*4+2, 1, "", tot_stay_fmt)
        ws.write(row+h*4+3, 0, h+1, title_fmt)
        ws.write(row+h*4+3, 1, "", row_title_fmt)
        ws.write(row+h*4+3, 2, "Stay", row_title_fmt)
        # Entry
        ws.write(row+h*4+4, 1, "", row_title_fmt)
        ws.write(row+h*4+4, 2, "Entry", row_title_fmt)

        # Calculating Total Stay for Hotel h
        total_stay = np.zeros(tot_days, dtype=int)
        for d in range(tot_days):
            total_stay[d] += stay_list[h, d]
        # End Calculation
        for d in range(tot_days):
            ws.write(row+h*4+1, 3+d, total_stay[d], tot_stay_fmt)
            if total_stay[d] > 0:
                ws.write(row+h*4+2, 3+d,
                         prob.hotel_capacity[h] - total_stay[d], remaining_fmt)
                vacancy[d] += prob.hotel_capacity[h] - total_stay[d]
            # Formating empty hotel days
            else:
                ws.write(row+h*4+2, 3+d, "", remaining_fmt)
                ws.write(row+h*4+3, 3+d, "", empty_hotel_fmt)

        # Merge Hotel Numbers
        # ws.merge_range(row+h*4+3, 0, row+h *
        #                3+3+15, 0, h+1, title_fmt)
        # for g in range(group_count):
        #     ws.write(row+h*3+3+g, 1, g+1, group_info_fmt)

        for d in range(tot_days):
            if d >= entry_days or entry_list[h, d] == 0:
                ws.write(row+h*4+4, 3+d, "", empty_hotel_fmt)

            if stay_list[h, d] < 1:
                continue
            if d < entry_days and entry_list[h, d] >= 1:
                ws.write(row+h*4+3, 3+d,
                         stay_list[h, d], entry_fmt)
                ws.write(row+h*4+4, 3+d, entry_list[h, d], entry_fmt)
            else:
                ws.write(row+h*4+3, 3+d,
                         stay_list[h, d], stay_fmt)

        ws.set_row(row+h*4+5, 2)
        row += 1

    # Filling vacancy row
    for i in range(tot_days):
        ws.write(4, 3+i, vacancy[i], row_title_fmt)
    ws.conditional_format(4, 3, 4, tot_days-1,
                          {'type': '2_color_scale', 'min_color': '#FFFD00', 'max_color': '#C00001'})

    workbook.close()
