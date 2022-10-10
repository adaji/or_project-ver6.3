from itertools import dropwhile
import numpy as np
import re
import xlsxwriter as xw
from xlsxwriter.utility import xl_rowcol_to_cell
import copy
import gurobipy as gp
from gurobipy import GRB
import json

opt_model = gp.Model(name="Allocate MIP Model")
optimal_takhsis = []
# Global vars for API response data
final_mec_groups = None
final_med2_groups = [[]]
med_first_tot_duration = []
med_first_group_num = []
med_last_tot_duration = []
med_last_group_num = []

downhaul_grouping_mfirst = None
backhaul_grouping_mlast = None

runtime = 0
objbst = 0
objbnd = 0
solcnt = 0
gap = 0.0


class ResProblem:
    def __init__(self,
                 tot_passengers=53622,
                 stay_duration=6,
                 tot_days=31,
                 hotel_price=[0.5, 0.6, 1, 0.6, 0.5, 0.6, 0.5, 0.8,
                              0.8, 0.5, 0.6, 0.6, 0.6, 0.6, 0.8, 0.6, 0.8, 0.6],
                 hotel_capacity=[819, 819, 1073, 514, 457, 667, 791, 486,
                                 1520, 1085, 1000, 892, 486, 463, 542, 597, 667, 429],
                 daily_flight_cap=[2213, 2124, 2186, 2186, 2065, 2194, 2199, 2203, 2205, 2002, 2194, 2033,
                                   2184, 2194, 2189, 2153, 1916, 2153, 2148, 1868, 2148, 2143, 2163, 2113, 1957, 564],
                 ) -> None:

        self.tot_passengers = tot_passengers
        self.stay_duration = stay_duration
        self.tot_days = tot_days
        self.hotel_price = hotel_price
        self.hotel_capacity = hotel_capacity
        self.daily_flight_cap = daily_flight_cap


class Takhsis:
    def __init__(self,
                 med_first_group=[[819, 452, 450, 495], [192, 791, 1085, 96], [624, 1520], [1000, 667, 336], [1056, 371, 667, 90], [48, 480, 892, 480], [729, 450, 463, 288], [90, 180, 791, 1085, 48], [405, 1520, 240], [228, 91, 1000, 667, 225], [1073, 372, 475, 192], [192, 892, 480, 528], [
                     729, 48, 457, 463, 278], [90, 186, 791, 1081, 45], [405, 1515, 270], [227, 94, 1000, 667], [1073, 360, 135, 420], [502, 892, 480], [718, 450, 463, 225], [45, 714, 1061, 93], [47, 1520, 279], [819, 96, 1000], [1073, 371, 405], [667, 892, 288], [774, 450, 45, 182, 450], [746, 1085]],
                 mec_first_group=[[2216], [2164], [187, 1957], [2003], [609, 1575], [1900], [1069, 861], [2194], [1511, 654], [2211], [76, 2036], [672, 1420], [1364, 611], [
                     2175, 18], [2190], [535, 1453], [1504, 484], [1874], [552, 1304], [1596, 317], [1846], [737, 1178], [1748, 101], [1847], [901, 1000], [512, 456, 456, 407]],
                 downhaul_grouping_mfirst=[],
                 med_last_group=[[986, 480, 450], [480, 1085, 144, 429], [819, 791, 460], [818, 360, 892], [1073, 238, 667], [276, 96, 1520], [990, 466, 450], [480, 1035, 381], [765, 791, 360], [
                     819, 96, 892, 96], [1073, 181, 667], [514, 180, 1520], [45, 1000, 480, 597, 45], [45, 1035, 667, 384], [765, 791, 542], [811, 48, 892, 463], [1053, 667], [514, 407]],
                 mec_last_group=[[922, 994], [523, 1615], [860, 1210], [1363, 707], [1789, 189], [1892], [487, 1419], [1071, 825], [
                     1765, 151], [1903], [472, 1449], [1088, 1126], [1480, 687], [1701, 430], [2004, 94], [2214], [175, 1545], [921]],
                 backhaul_grouping_mlast=[],
                 items_value=[0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54,
                              0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36,
                              0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                              -1],
                 items_weight=[192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168,
                               144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126,
                               96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84,
                               1],

                 ) -> None:

        self.med_first_group = med_first_group
        self.mec_first_group = mec_first_group
        self.downhaul_grouping_mfirst = downhaul_grouping_mfirst
        self.med_last_group = med_last_group
        self.mec_last_group = mec_last_group
        self.backhaul_grouping_mlast = backhaul_grouping_mlast
        self.items_value = items_value
        self.items_weight = items_weight


def create_takhsis_model(prob=Takhsis()):
    global downhaul_grouping_mfirst
    global backhaul_grouping_mlast
    global opt_model

    opt_model = gp.Model(name="Allocate MIP Model")  # Make clean model

    downhaul_grouping_mfirst_index = []
    backhaul_grouping_mlast_index = []

    first_days = len(prob.med_first_group)  # 26
    last_days = len(prob.med_last_group)  # 18

    num_weights = len(prob.items_weight)

    first_packs = []  # size will be 28
    for d in range(first_days):
        downhaul_grouping_mfirst_index.append(downhaul_grouping_mfirst[d, :].nonzero()[0].tolist())
        first_packs.append([prob.med_first_group[d], prob.mec_first_group[d],
                           downhaul_grouping_mfirst[d, :][downhaul_grouping_mfirst[d, :].nonzero()].tolist()])

    last_packs = []  # size will be 18
    for d in range(last_days):
        # print(d)
        # print(list(backhaul_grouping_mlast[d, :][backhaul_grouping_mlast[d, :].nonzero()]))
        backhaul_grouping_mlast_index.append(backhaul_grouping_mlast[d, :].nonzero()[0].tolist())
        last_packs.append([prob.med_last_group[d], prob.mec_last_group[d],
                          list(backhaul_grouping_mlast[d, :][backhaul_grouping_mlast[d, :].nonzero()])])

    # Remove print
    # print("downhaul grouping mfirst-index non zero-days-flight station")
    # print(downhaul_grouping_mfirst_index)
    # print("backhaul grouping mlast-index non zero-days-flight station")
    # print(backhaul_grouping_mlast_index)
    # REmove above

    # Writing downhaul_grouping_mfirst_index and backhaul to file for allocate_mec use
    with open("./Output/Takhsis/downhaul_grouping_mfirst_index", "w") as fp:
        json.dump(downhaul_grouping_mfirst_index, fp)
    with open("./Output/Takhsis/backhaul_grouping_mlast_index", "w") as fp:
        json.dump(backhaul_grouping_mlast_index, fp)

    packs = first_packs + last_packs  # size will be 28 + 18

    num_packs = 3  # len(packs)  # Number of packs 3
    num_items_per_pack = []
    for d in range(first_days+last_days):
        num_items_per_pack.append([len(i) for i in packs[d]])

    # Small packs should be joined with bigger packs

    x = {(d, w, p, i): opt_model.addVar(vtype=GRB.INTEGER, lb=0, name='X(%i_%i_%i_%i)' % (d, w, p, i))
         for d in range(first_days+last_days)
         for w in range(num_weights) for p in range(num_packs) for i in range(num_items_per_pack[d][p])}

    opt_model.addConstrs(gp.quicksum(x[d, w, p, i] * prob.items_weight[w]
                                     for w in range(num_weights)) == packs[d][p][i]
                         for d in range(first_days+last_days) for p in range(num_packs) for i in range(num_items_per_pack[d][p]))

    # I guess below is redundant - the above constraint could suffice
    # opt_model.addConstrs(gp.quicksum(x[d, w, p, i] * prob.items_weight[w]
    #                                  for i in range(num_items_per_pack[d][p]) for w in range(num_weights)) == sum(packs[d][p])
    #                      for d in range(first_days+last_days) for p in range(num_packs))

    # Each bin final solution must be equal to other bins
    for p in range(1, num_packs):
        opt_model.addConstrs(gp.quicksum(x[d, w, 0, i] for i in range(num_items_per_pack[d][0]))
                             == gp.quicksum(x[d, w, p, i] for i in range(num_items_per_pack[d][p]))
                             for w in range(num_weights) for d in range(first_days+last_days))

    # Old objective
    # obj = gp.quicksum(prob.items_value[w]*x[d, w, p, i]
    #                   for d in range(first_days+last_days) for w in range(num_weights) for p in range(num_packs) for i in range(num_items_per_pack[d][p]))

    # opt_model.ModelSense = GRB.MAXIMIZE
    # opt_model.setObjective(obj)
     
    # New Objective - Handling a target percentage for each 45 multiple
    

    # opt_model.write("out-param.lp")
    # opt_model.write("out-param.mps")

    # Model Parameters
    opt_model.Params.timelimit = 10000
    opt_model.Params.integralityfocus = 1
    # opt_model.Params.seed = 123
    # opt_model.Params.mipfocus = 1

    def getSol(model, where):
        if where == GRB.Callback.MIPSOL:
            global optimal_takhsis
            global solcnt

            optimal_takhsis = model.cbGetSolution(x)
            solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)

        elif where == GRB.Callback.MIP:
            global runtime
            global objbst
            global objbnd
            global gap

            runtime = model.cbGet(GRB.Callback.RUNTIME)
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            gap = (objbnd-objbst)/objbst

    opt_model._vars = opt_model.getVars()
    opt_model.optimize(getSol)


def build_mec_groups(group1, group2):
    g1 = copy.deepcopy(group1)
    g2 = copy.deepcopy(group2)
    # print("G1:")
    # print(g1)
    # print("G2:")
    # print(g2)
    # Grouping
    new_g = [[] for i in range(len(g1))]
    d1 = 0
    for i in range(len(g2)):
        bin2 = g2[i]

        for j in range(d1, len(g1)):
            bin1 = g1[j]
            if bin1 <= bin2:
                new_g[j].append(bin1)
                bin2 -= bin1
                d1 = j+1
                continue
            else:
                new_g[j].append(bin2)
                g1[j] -= bin2
                break

    # print('Main Groups')
    # for i in range(len(new_g)):
    #     print(new_g[i])
    main_groups = copy.deepcopy(new_g)
    # print("********")

    # for grups smaller than 90-3=87, join to other groups
    for i in range(len(new_g)):
        for j in range(len(new_g[i])):
            # print(i, j)
            if new_g[i][j] < 87:
                if j < len(new_g[i]) - 1:
                    # join with next
                    new_g[i][j+1] += new_g[i][j]
                    new_g[i].pop(j)
                    break
                else:
                    # join with previous
                    new_g[i][j-1] += new_g[i][j]
                    new_g[i].pop(j)
                    break

    # print('Final groups binding small groups to larger ones')
    # for i in range(len(new_g)):
    #     print(new_g[i])

    # print("Build_MEC Output")
    # print("Main Groups")
    # print(main_groups)
    # print("New G")
    # print(new_g)
    return main_groups, new_g


def visualize(
    prob1,
    prob2,
    med_first_backhaul_start_day,
    med_last_downhaul_start_day,
    med_last_backhaul_start_day,
    jeddah_flights_med_first,
    jeddah_flights_med_last,
    med_first_res_solution_x,
    med_first_res_solution_y,
    med_last_res_solution_x,
    med_last_res_solution_y,
    items_weight,
    items_value,
    excel_output,
):
    # # For med first solution
    # with open(prob1.solution_path) as f:
    #     hotel_count = len(prob1.hotel_capacity)
    #     content = f.read()

    #     entry1 = re.findall(r'X\((\d+)_(\d+)\) (\d+)', content)

    #     entry_list1 = np.zeros(
    #         (hotel_count, prob1.tot_days - prob1.stay_duration+1), dtype=int)

    #     for x in entry1:
    #         # Remove -1 if using optimal solution - -1 just for X0
    #         entry_list1[int(x[0]), int(x[1])] = int(x[2])

    #     # **************************

    #     stay1 = re.findall(r'Y\((\d+)_(\d+)\) 1', content)

    #     hotel_filled1 = np.zeros((hotel_count, prob1.tot_days), dtype=int)
    #     for y in stay1:
    #         hotel_filled1[int(y[0]), int(y[1])] = 1
    hotel_count = len(prob1.hotel_capacity)
    entry_list1 = np.array(med_first_res_solution_x)
    hotel_filled1 = np.array(med_first_res_solution_y)
    # Creating data for schedule table
    # Create stay data
    stay_list1 = np.zeros((hotel_count, prob1.tot_days), dtype=int)
    for h in range(hotel_count):
        for d in range(prob1.tot_days):
            for i in range(max(d-prob1.stay_duration+1, 0), min(d+1, prob1.tot_days-prob1.stay_duration+1)):
                stay_list1[h, d] += entry_list1[h, i]
                # If in input data we don't have Y[h,d], we can calculate it by uncommenting below
                # if stay_list[h, d] > 0:
                #     hotel_filled[h, d] = 1

    # Calculating objective function value for current solution
    objective_value1 = 0

    for h in range(hotel_count):
        hotel_total_filled = 0
        for d in range(prob1.tot_days):
            hotel_total_filled += hotel_filled1[h, d]
        objective_value1 += hotel_total_filled * \
            prob1.hotel_capacity[h] * prob1.hotel_price[h]

    print("Objective value 1 for current solution: %i" % (objective_value1))

    # For med last solution

    # with open(prob2.solution_path) as f:
    #     hotel_count = len(prob2.hotel_capacity)
    #     content = f.read()

    #     entry2 = re.findall(r'X\((\d+)_(\d+)\) (\d+)', content)

    #     entry_list2 = np.zeros(
    #         (hotel_count, prob2.tot_days - prob2.stay_duration+1), dtype=int)

    #     for x in entry2:
    #         # Remove -1 if using optimal solution - -1 just for X0
    #         entry_list2[int(x[0]), int(x[1])] = int(x[2])

    #     # **************************

    #     stay2 = re.findall(r'Y\((\d+)_(\d+)\) 1', content)

    #     hotel_filled2 = np.zeros((hotel_count, prob2.tot_days), dtype=int)
    #     for y in stay2:
    #         hotel_filled2[int(y[0]), int(y[1])] = 1

    entry_list2 = np.array(med_last_res_solution_x)
    hotel_filled2 = np.array(med_last_res_solution_y)
    # Creating data for schedule table
    # Create stay data
    stay_list2 = np.zeros((hotel_count, prob2.tot_days), dtype=int)
    for h in range(hotel_count):
        for d in range(prob2.tot_days):
            for i in range(max(d-prob2.stay_duration+1, 0), min(d+1, prob2.tot_days-prob2.stay_duration+1)):
                stay_list2[h, d] += entry_list2[h, i]
                # If in input data we don't have Y[h,d], we can calculate it by uncommenting below
                # if stay_list[h, d] > 0:
                #     hotel_filled[h, d] = 1

    # Calculating objective function value for current solution
    objective_value2 = 0

    for h in range(hotel_count):
        hotel_total_filled = 0
        for d in range(prob2.tot_days):
            hotel_total_filled += hotel_filled2[h, d]
        objective_value2 += hotel_total_filled * \
            prob2.hotel_capacity[h] * prob2.hotel_price[h]

    print("Objective value 2 for current solution: %i" % (objective_value2))

    # Creating spreadsheet
    workbook = xw.Workbook(excel_output, {'in_memory': True})
    ws = workbook.add_worksheet("Residence Plan")  # ws means worksheet

    # Before writing values, define formatting
    blank_fmt = workbook.add_format({'bg_color': '#FFFFFF'})
    obj_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#003296', 'font_color': '#FFFFFF', 'bold': True})
    tot_entry_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#ED7D31', 'align': 'center', 'bold': True})
    tot_entry_left_fmt = workbook.add_format(
        {'num_format': '#,##0', 'bg_color': '#ED7D31', 'align': 'left', 'bold': True})
    tot_entry_diff_fmt = workbook.add_format(
        {'italic': True, 'bg_color': '#F4B084', 'align': 'center', 'bold': True})
    tot_entry_diff_left_fmt = workbook.add_format(
        {'italic': True, 'bg_color': '#F4B084', 'align': 'left', 'bold': True})
    title_fmt = workbook.add_format(
        {'bg_color': '#0070C0', 'font_color': '#FFFFFF', 'align': 'center', 'valign': 'vcenter', 'bold': True})
    capacity_fmt = workbook.add_format(
        {'bg_color': '#00B0F0', 'align': 'center', 'bold': True})
    row_title_fmt = workbook.add_format(
        {'bg_color': '#FFC000', 'align': 'left', 'num_format': '#,##0', })
    tot_stay_title_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'align': 'right', 'bold': True})
    tot_stay_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'align': 'center', 'bold': True})
    remaining_fmt = workbook.add_format(
        {'bg_color': '#B4C6E7', 'font_color': 'red', 'align': 'center', 'bold': True})
    entry_fmt = workbook.add_format({'bg_color': '#00B050', 'align': 'center'})
    stay_fmt = workbook.add_format({'bg_color': '#92D050', 'align': 'center'})
    empty_hotel_fmt = workbook.add_format({'bg_color': '#D9D9D9'})
    header_fmt = workbook.add_format(
        {'align': 'center', 'valign': 'vcenter', 'bold': True, 'bg_color': '#002060', 'font_color': '#FFFFFF', 'font_size': 20})
    date_fmt = workbook.add_format({'bg_color': '#31869B', 'font_color': '#FFFFFF',
                                   'font_size': 12, 'align': 'center', 'valign': 'vcenter', 'bold': True, 'rotation': 90, 'num_format': '[$-ar-SA,286]B2d mmmm yyyy;@'})
    mec_group_fmt = workbook.add_format(
        {'bg_color': '#7030A0', 'font_color': '#FFFFFF', 'align': 'center'})
    mec_group_title_fmt = workbook.add_format(
        {'bg_color': '#FFFF00', 'align': 'center'})
    duration_fmt = workbook.add_format(
        {'bg_color': '#D9D9D9', 'font_color': '#FF0000', 'bold': True, 'align': 'center'})
    mec_title_fmt = workbook.add_format(
        {'bg_color': '#92D050', 'bold': True, 'num_format': '#,##0'})
    mec_entry_fmt = workbook.add_format(
        {'bg_color': '#00B0F0', 'bold': True, 'num_format': '#,##0', 'align': 'center'})
    tot_mec_stay_fmt = workbook.add_format(
        {'bg_color': '#EC9AA0', 'font_size': 8, 'num_format': '#,##0', 'align': 'center'})

    # End formatting definition
    # Making white background
    [ws.write(i, j, "", blank_fmt) for i in range(400) for j in range(150)]
    ws.set_zoom(55)
    ws.freeze_panes(7, 3)
    ws.set_column(3, 3+prob2.tot_days+med_first_backhaul_start_day, 4.83)
    ws.write(1, 0, "Objective Value 1:", obj_fmt)
    ws.write(1, 1, "", obj_fmt)
    ws.write(1, 2, objective_value1, obj_fmt)
    for d in range(med_first_backhaul_start_day+prob2.tot_days):
        ws.write(1, 3+d, "", blank_fmt)
    ws.write(1, med_first_backhaul_start_day, "Objective Value 2:", obj_fmt)
    ws.write(1, med_first_backhaul_start_day+1, "", obj_fmt)
    ws.write(1, med_first_backhaul_start_day+2, "", obj_fmt)
    ws.merge_range(1, med_first_backhaul_start_day+3, 1,
                   med_first_backhaul_start_day+4, objective_value2, obj_fmt)

    # Total Entry Row
    # Calculation
    total_entry1 = np.zeros(prob1.tot_days, dtype=int)
    for d in range(prob1.tot_days-prob1.stay_duration+1):
        for h in range(hotel_count):
            total_entry1[d] += entry_list1[h, d]
    # Med Last
    total_entry2 = np.zeros(prob2.tot_days, dtype=int)
    for d in range(prob2.tot_days-prob2.stay_duration+1):
        for h in range(hotel_count):
            total_entry2[d] += entry_list2[h, d]
    # End Calculation

    ws.write(2, 0, 'Total Entry', tot_entry_left_fmt)
    ws.write(2, 1, '', tot_entry_fmt)
    ws.write(2, 2, '', tot_entry_fmt)
    for i in range(prob1.tot_days):
        ws.write(2, 3+i, total_entry1[i], tot_entry_fmt)

    ws.write(3, 0, 'Total Expected Entry', tot_entry_left_fmt)
    ws.write(3, 1, '', tot_entry_fmt)
    ws.write(3, 2, '', tot_entry_fmt)
    for i in range(prob1.tot_days):
        if i <= prob1.tot_days-prob1.stay_duration:
            ws.write(3, 3+i, prob1.daily_flight_cap[i], tot_entry_fmt)
        else:
            ws.write(3, 3+i, "", tot_entry_fmt)

    ws.write(4, 0, 'Difference', tot_entry_diff_left_fmt)
    ws.write(4, 1, '', tot_entry_diff_fmt)
    ws.write(4, 2, '', tot_entry_diff_fmt)
    for i in range(prob1.tot_days):
        if i <= prob1.tot_days-prob1.stay_duration:
            ws.write(
                4, 3+i, prob1.daily_flight_cap[i]-total_entry1[i], tot_entry_diff_fmt)
        else:
            ws.write(4, 3+i, "", tot_entry_diff_fmt)

    # Total Vacancy
    ws.write(5, 0, 'Vacancy', row_title_fmt)
    ws.write(5, 1, '', row_title_fmt)
    ws.write(5, 2, '', row_title_fmt)
    vacancy1 = np.zeros(prob1.tot_days, dtype=int)
    # will be filled at the bottom of program

    # All above same for med last prob2
    ws.write(2, med_first_backhaul_start_day,
             'Total Entry', tot_entry_left_fmt)
    ws.write(2, med_first_backhaul_start_day+1, '', tot_entry_fmt)
    ws.write(2, med_first_backhaul_start_day+2, '', tot_entry_fmt)
    for i in range(prob2.tot_days):
        ws.write(2, 3+med_first_backhaul_start_day +
                 i, total_entry2[i], tot_entry_fmt)

    ws.write(3, med_first_backhaul_start_day+0,
             'Total Expected Entry', tot_entry_left_fmt)
    ws.write(3, med_first_backhaul_start_day+1, '', tot_entry_fmt)
    ws.write(3, med_first_backhaul_start_day+2, '', tot_entry_fmt)
    for i in range(prob2.tot_days):
        if i <= prob2.tot_days-prob2.stay_duration:
            ws.write(3, 3+med_first_backhaul_start_day+i,
                     prob2.daily_flight_cap[i], tot_entry_fmt)
        else:
            ws.write(3, 3+med_first_backhaul_start_day+i, "", tot_entry_fmt)

    ws.write(4, med_first_backhaul_start_day+0,
             'Difference', tot_entry_diff_left_fmt)
    ws.write(4, med_first_backhaul_start_day+1, '', tot_entry_diff_fmt)
    ws.write(4, med_first_backhaul_start_day+2, '', tot_entry_diff_fmt)
    for i in range(prob2.tot_days):
        if i <= prob2.tot_days-prob2.stay_duration:
            ws.write(
                4, 3+med_first_backhaul_start_day+i, prob2.daily_flight_cap[i]-total_entry2[i], tot_entry_diff_fmt)
        else:
            ws.write(4, 3+med_first_backhaul_start_day +
                     i, "", tot_entry_diff_fmt)

    # Total Vacancy
    ws.write(5, med_first_backhaul_start_day+0, 'Vacancy', row_title_fmt)
    ws.write(5, med_first_backhaul_start_day+1, '', row_title_fmt)
    ws.write(5, med_first_backhaul_start_day+2, '', row_title_fmt)
    vacancy2 = np.zeros(prob2.tot_days, dtype=int)
    # will be filled at the bottom of program

    # Header Row - Number of Days
    ws.write(6, 0, "Hotel \ Day", title_fmt)
    ws.write(6, 1, "", title_fmt)
    ws.write(6, 2, "", title_fmt)

    for i in range(med_first_backhaul_start_day+prob2.tot_days):
        ws.write(6, 3+i, i+1, title_fmt)
        cell = xl_rowcol_to_cell(0, 2+i)
        ws.write_formula(0, 3+i, "=" + cell + "+1", date_fmt)

    date_first_fmt = workbook.add_format({'bg_color': '#31869B', 'font_color': '#FFFF00',
                                          'font_size': 12, 'align': 'center', 'valign': 'vcenter', 'bold': True, 'rotation': 90, 'num_format': '[$-ar-SA,286]B2d mmmm yyyy;@'})
    ws.write(0, 3, "1443/11/5", date_first_fmt)
    ws.set_row(0, 105)  # Height of row 0 - date
    # MEDINAH
    ws.set_row(7, 37)
    ws.merge_range(7, 0, 7, 3+prob1.tot_days-1, "مدینه - قبل", header_fmt)
    ws.merge_range(7, 3+prob1.tot_days, 7, 3 +
                   med_first_backhaul_start_day-1, "", header_fmt)
    ws.merge_range(7, 3+med_first_backhaul_start_day, 7, 3 +
                   med_first_backhaul_start_day+prob2.tot_days-1, "مدینه - بعد", header_fmt)
    # Data Cells
    row = 7
    col = 0
    for h in range(hotel_count):
        # Header Row - Total Stay & Remaining Cap
        ws.write(row+h*4+1, 0, "Capacity", capacity_fmt)
        ws.write(row+h*4+2, 0,
                 prob1.hotel_capacity[h], capacity_fmt)

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
        total_stay1 = np.zeros(prob1.tot_days, dtype=int)
        for d in range(prob1.tot_days):
            total_stay1[d] += stay_list1[h, d]

        # Med Last
        total_stay2 = np.zeros(prob2.tot_days, dtype=int)
        for d in range(prob2.tot_days):
            total_stay2[d] += stay_list2[h, d]
        # End Calculation

        for d in range(prob1.tot_days):
            ws.write(row+h*4+1, 3+d, total_stay1[d], tot_stay_fmt)
            if total_stay1[d] > 0:
                ws.write(row+h*4+2, 3+d,
                         prob1.hotel_capacity[h] - total_stay1[d], remaining_fmt)
                vacancy1[d] += prob1.hotel_capacity[h] - total_stay1[d]
            # Formating empty hotel days
            else:
                ws.write(row+h*4+2, 3+d, "", remaining_fmt)
                ws.write(row+h*4+3, 3+d, "", empty_hotel_fmt)
        # Above for same med last
        for d in range(prob2.tot_days):
            ws.write(row+h*4+1, 3+med_first_backhaul_start_day +
                     d, total_stay2[d], tot_stay_fmt)
            if total_stay2[d] > 0:
                ws.write(row+h*4+2, 3+med_first_backhaul_start_day+d,
                         prob2.hotel_capacity[h] - total_stay2[d], remaining_fmt)
                vacancy2[d] += prob2.hotel_capacity[h] - total_stay2[d]
            # Formating empty hotel days
            else:
                ws.write(row+h*4+2, 3+med_first_backhaul_start_day +
                         d, "", remaining_fmt)
                ws.write(row+h*4+3, 3+med_first_backhaul_start_day +
                         d, "", empty_hotel_fmt)

        # Merge Hotel Numbers
        # ws.merge_range(row+h*4+3, 0, row+h *
        #                3+3+15, 0, h+1, title_fmt)
        # for g in range(group_count):
        #     ws.write(row+h*3+3+g, 1, g+1, group_info_fmt)

        for d in range(prob1.tot_days):
            if d >= prob1.tot_days-prob1.stay_duration+1 or entry_list1[h, d] == 0:
                ws.write(row+h*4+4, 3+d, "", empty_hotel_fmt)

            if stay_list1[h, d] < 1:
                continue
            if d < prob1.tot_days-prob1.stay_duration+1 and entry_list1[h, d] >= 1:
                ws.write(row+h*4+3, 3+d,
                         stay_list1[h, d], entry_fmt)
                ws.write(row+h*4+4, 3+d, entry_list1[h, d], entry_fmt)
            else:
                ws.write(row+h*4+3, 3+d,
                         stay_list1[h, d], stay_fmt)

        # Above for med last
        for d in range(prob2.tot_days):
            if d >= prob2.tot_days-prob2.stay_duration+1 or entry_list2[h, d] == 0:
                ws.write(row+h*4+4, 3+med_first_backhaul_start_day +
                         d, "", empty_hotel_fmt)

            if stay_list2[h, d] < 1:
                continue
            if d < prob2.tot_days-prob2.stay_duration+1 and entry_list2[h, d] >= 1:
                ws.write(row+h*4+3, 3+med_first_backhaul_start_day+d,
                         stay_list2[h, d], entry_fmt)
                ws.write(row+h*4+4, 3+med_first_backhaul_start_day +
                         d, entry_list2[h, d], entry_fmt)
            else:
                ws.write(row+h*4+3, 3+med_first_backhaul_start_day+d,
                         stay_list2[h, d], stay_fmt)

        ws.set_row(row+h*4+5, 2)
        row += 1

    # Filling vacancy row
    for i in range(prob1.tot_days):
        ws.write(5, 3+i, vacancy1[i], row_title_fmt)
    ws.conditional_format(5, 3, 5, prob1.tot_days-1,
                          {'type': '2_color_scale', 'min_color': '#FFFD00', 'max_color': '#C00001'})

    # Above for med last
    for i in range(prob2.tot_days):
        ws.write(5, 3+med_first_backhaul_start_day +
                 i, vacancy2[i], row_title_fmt)
    ws.conditional_format(5, 3+med_first_backhaul_start_day, 5, med_first_backhaul_start_day+prob2.tot_days-1,
                          {'type': '2_color_scale', 'min_color': '#FFFD00', 'max_color': '#C00001'})

    # Medina Residence plan based on medina flights completed
    # Now building overview of medina & mecca altogether

    row = 5*hotel_count + 9
    ws.merge_range(row, 0, row, 3+med_first_backhaul_start_day +
                   prob2.tot_days-1, 'مکه', header_fmt)
    ws.set_row(row, 37)
    row += 1
    ws.merge_range(row, 0, row, 3+med_first_backhaul_start_day +
                   prob2.tot_days-1, 'مدینه - قبل', header_fmt)
    ws.set_row(row, 33)
    row += 1

    # Day number
    for i in range(prob1.tot_days-prob1.stay_duration+1):
        ws.write(row, 3+i+prob1.stay_duration, i+1, mec_group_fmt)
        ws.write(row+1, 3+i+prob1.stay_duration,
                 total_entry1[i], tot_entry_fmt)
    for i in range(len(jeddah_flights_med_first)):
        ws.write(row, 3+med_first_backhaul_start_day+i, i+1, title_fmt)
        ws.write(row+1, 3+med_first_backhaul_start_day+i,
                 jeddah_flights_med_first[i], mec_title_fmt)

    row += 1
    ws.write(row, 0, 'طول سفر', mec_group_title_fmt)
    ws.write(row, 1, 'روز ورود', mec_group_title_fmt)
    ws.write(row, 2, 'گروه', mec_group_title_fmt)
    row += 1

    # Running Build Mecca Groups Algorithm for Med First
    global final_mec_groups
    main_mec_groups, final_mec_groups = build_mec_groups(
        total_entry1[:-prob1.stay_duration+1].tolist(), jeddah_flights_med_first)

    # Filling Medina First
    global med_first_tot_duration
    global med_first_group_num
    mec_exit_day = 0
    jed_stay = jeddah_flights_med_first.copy()
    for i in range(len(main_mec_groups)):
        for j in range(len(main_mec_groups[i])):
            # Duration of each group
            med_first_tot_duration.append(med_first_backhaul_start_day + mec_exit_day-i)
            ws.write(row, 0, med_first_tot_duration[-1], duration_fmt)
            med_first_group_num.append([i+1, j+1])
            ws.write(row, 1, i+1, mec_group_fmt)
            ws.write(row, 2, j+1, mec_group_fmt)
            # iran green 1
            for d in range(3, 3+i):
                ws.write(row, d, "", mec_title_fmt)
            # Build medina entries and stays
            ws.write(row, 3+i, main_mec_groups[i][j], mec_entry_fmt)
            ws.write(row, 3+i+prob1.stay_duration,
                     main_mec_groups[i][j], mec_entry_fmt)
            # orange
            for d in range(prob1.stay_duration-1):
                ws.write(row, 4+i+d, "", tot_entry_fmt)
            # blue mec
            for d in range(3+i+prob1.stay_duration+1, 3+med_first_backhaul_start_day+mec_exit_day):
                ws.write(row, d, "", title_fmt)
            # iran green
            for d in range(3+med_first_backhaul_start_day+mec_exit_day+1, 3+med_first_backhaul_start_day+len(jeddah_flights_med_first)):
                ws.write(row, d, "", mec_title_fmt)
            # mec exit
            ws.write(row, 3+med_first_backhaul_start_day+mec_exit_day,
                     main_mec_groups[i][j], mec_entry_fmt)
            jed_stay[mec_exit_day] -= main_mec_groups[i][j]
            if jed_stay[mec_exit_day] == 0:
                mec_exit_day += 1

            row += 1

    # Mecca total formula for charting
    # Calculation
    med_first_mec_entry_total = [0]*(med_first_backhaul_start_day+len(jeddah_flights_med_first))
    med_first_mec_entry_total[prob1.stay_duration:prob1.stay_duration +
                              len(prob1.daily_flight_cap)] = np.cumsum(prob1.daily_flight_cap).tolist()
    # keep last med first entry to the end
    for i in range(med_first_backhaul_start_day-prob1.tot_days):
        med_first_mec_entry_total[prob1.stay_duration +
                                  len(prob1.daily_flight_cap) + i] = sum(prob1.daily_flight_cap)

    # subtracting med first returning to iran
    jeddah_flights_med_first_total = np.cumsum(jeddah_flights_med_first).tolist()
    for i in range(len(jeddah_flights_med_first)):
        med_first_mec_entry_total[med_first_backhaul_start_day +
                                  i] = sum(prob1.daily_flight_cap) - jeddah_flights_med_first_total[i]

    # writing to excel
    first_date_fmt = workbook.add_format({'font_color': '#FF0000', 'align': 'right', 'bold': True})
    ws.write(row, 2, "Total Med First", first_date_fmt)
    for i in range(len(med_first_mec_entry_total)):
        ws.write(row, 3+i, med_first_mec_entry_total[i], tot_mec_stay_fmt)
    tot_med_first_row = row

    # Filling Medinah Last
    ws.merge_range(row+2, 0, row+2, 3+med_first_backhaul_start_day +
                   prob2.tot_days-1, 'مدینه - بعد', header_fmt)
    ws.set_row(row+2, 33)
    row += 3

    # Day number
    for i in range(len(jeddah_flights_med_last)):
        ws.write(row, 2+med_last_downhaul_start_day+i, i+1, mec_group_fmt)
        ws.write(row+1, 2+med_last_downhaul_start_day+i,
                 jeddah_flights_med_last[i], mec_title_fmt)
    for i in range(prob2.tot_days-prob2.stay_duration+1):
        ws.write(row, 2+med_first_backhaul_start_day +
                 (med_last_backhaul_start_day+1)+i, i+1, title_fmt)
        ws.write(row+1, 2+med_first_backhaul_start_day + (med_last_backhaul_start_day+1)+i,
                 total_entry2[i], tot_entry_fmt)

    row += 1
    ws.write(row, 0, 'طول سفر', mec_group_title_fmt)
    ws.write(row, 1, 'روز ورود', mec_group_title_fmt)
    ws.write(row, 2, 'گروه', mec_group_title_fmt)
    row += 1

    # Running Build Mecca Groups Algorithm for Med Last
    main_mec2_groups, final_mec2_groups = build_mec_groups(
        jeddah_flights_med_last, prob2.daily_flight_cap)  # total_entry2[:-prob2.stay_duration+1].tolist())
    global final_med2_groups
    global med_last_tot_duration
    global med_last_group_num
    med_exit_day = 0
    med_stay = prob2.daily_flight_cap.copy()  # total_entry2[:-prob2.stay_duration+1].copy()
    for i in range(len(main_mec2_groups)):
        for j in range(len(main_mec2_groups[i])):
            # Duration of each group
            med_last_tot_duration.append(
                med_first_backhaul_start_day + (med_last_backhaul_start_day+1)+med_exit_day-(i+med_last_downhaul_start_day)+1)
            ws.write(row, 0, med_first_tot_duration[-1], duration_fmt)
            med_last_group_num.append([i+1, j+1])
            ws.write(row, 1, i+1, mec_group_fmt)
            ws.write(row, 2, j+1, mec_group_fmt)
            # iran green 1
            for d in range(3, 2+i+med_last_downhaul_start_day):
                ws.write(row, d, "", mec_title_fmt)
            # Build mecca entries
            ws.write(row, 2+i+med_last_downhaul_start_day,
                     main_mec2_groups[i][j], mec_entry_fmt)

            # blue jed
            for d in range(3+i+med_last_downhaul_start_day, 2+med_first_backhaul_start_day +
                           (med_last_backhaul_start_day+1)+med_exit_day-prob2.stay_duration):
                ws.write(row, d, "", title_fmt)

            # orange
            for d in range(3+med_first_backhaul_start_day +
                           (med_last_backhaul_start_day+1)-prob2.stay_duration+med_exit_day, 2+med_first_backhaul_start_day +
                           (med_last_backhaul_start_day+1)+med_exit_day):
                ws.write(row, d, "", tot_entry_fmt)

            # iran green 2
            for d in range(2+med_first_backhaul_start_day +
                           (med_last_backhaul_start_day+1)+med_exit_day+1, 2+med_first_backhaul_start_day +
                           (med_last_backhaul_start_day+1)+len(prob2.daily_flight_cap)):
                ws.write(row, d, "", mec_title_fmt)
            # med exit
            # Build medinah entries
            ws.write(row, 2+med_first_backhaul_start_day +
                     (med_last_backhaul_start_day+1)+med_exit_day,
                     main_mec2_groups[i][j], mec_entry_fmt)  # Exit
            ws.write(row, 2+med_first_backhaul_start_day +
                     (med_last_backhaul_start_day+1)-prob2.stay_duration+med_exit_day, main_mec2_groups[i][j], mec_entry_fmt)
            med_stay[med_exit_day] -= main_mec2_groups[i][j]
            final_med2_groups[med_exit_day].append(main_mec2_groups[i][j])
            # print(i, j, "appended")
            if med_stay[med_exit_day] == 0:
                med_exit_day += 1
                final_med2_groups.append([])

            row += 1
        # print("med_exit_day")
        # print(med_exit_day)
        # print("med_stay")
        # print(med_stay)
        # print("final_med2_groups")
        # print(final_med2_groups)

    # print("main_mec2_groups")
    # print(main_mec2_groups)
    # print("final_main_mec2_groups")
    # print(final_mec2_groups)

    # Mecca total formula for charting for Med last
    # Calculation
    med_last_mec_entry_total = [0]*(med_first_backhaul_start_day +
                                    (med_last_backhaul_start_day+1)+len(prob2.daily_flight_cap)-prob2.stay_duration)
    med_last_mec_entry_total[med_last_downhaul_start_day-1:med_last_downhaul_start_day +
                             len(jeddah_flights_med_last)-1] = np.cumsum(jeddah_flights_med_last).tolist()

    # keep last med first entry to the beginning of backhaul
    for i in range(med_first_backhaul_start_day + (med_last_backhaul_start_day+1)-len(jeddah_flights_med_last)-med_last_downhaul_start_day):
        med_last_mec_entry_total[med_last_downhaul_start_day - 1 +
                                 len(jeddah_flights_med_last) + i] = sum(jeddah_flights_med_last)

    # subtracting med first returning to iran
    med_flights_med_last_total = np.cumsum(prob2.daily_flight_cap).tolist()
    for i in range(len(prob2.daily_flight_cap)):
        med_last_mec_entry_total[med_first_backhaul_start_day + (med_last_backhaul_start_day+1) - prob2.stay_duration + i] = sum(
            jeddah_flights_med_last) - med_flights_med_last_total[i]

    # writing to excel
    ws.write(row, 2, "Total Med Last", first_date_fmt)
    for i in range(len(med_last_mec_entry_total)):
        ws.write(row, 3+i, med_last_mec_entry_total[i], tot_mec_stay_fmt)
    tot_med_last_row = row
    # BEFORE ANYTHING ELSE: Run takhsis optimization model
    # Med-first relates to residence solution for medina in med-first
    med_first_group = [[j for j in i if j > 0]
                       for i in entry_list1.transpose().tolist()]  # med-first entries
    med_first_group_index = [[j for j in range(len(i)) if i[j] > 0]
                             for i in entry_list1.transpose().tolist()]  # med-first entriies -keeping index for later in mecca assignment

    med_last_group = [[j for j in i if j > 0]
                      for i in entry_list2.transpose().tolist()]  # med-last entries
    med_last_group_index = [[j for j in range(len(i)) if i[j] > 0]
                            for i in entry_list2.transpose().tolist()]  # med-last entries -keeping index for later in mecca assignment

    # Writing med_first_group_index and med_last to file for alloc_mec use
    with open("./Output/Takhsis/med_first_group_index", "w") as fp:
        json.dump(med_first_group_index, fp)
    with open("./Output/Takhsis/med_last_group_index", "w") as fp:
        json.dump(med_last_group_index, fp)

    # print("Optimization INPUT Data")
    # print('med first group-eskan index')
    # print(med_first_group)
    # print(med_first_group_index)
    # print("med_last_group-eskan index")
    # print(med_last_group)
    # print(med_last_group_index)
    # # print('final mec grouo')
    # # print(final_mec_groups)

    # # for i in range(len(med_first_group)):
    # #     print(sum(med_first_group[i]), sum(final_mec_groups[i]))

    # print("med_last_group")
    # print(med_last_group)
    # print(len(med_last_group))
    # print("final_med2_groups")
    # Last item is []
    final_med2_groups.pop(-1)
    old_final_med2_groups = copy.deepcopy(final_med2_groups)
    # Bind smaller groups to large ones
    for i in range(len(final_med2_groups)):
        for j in range(len(final_med2_groups[i])):
            if final_med2_groups[i][j] < 84:
                if j < len(final_med2_groups[i]) - 1:
                    # join with next
                    final_med2_groups[i][j+1] += final_med2_groups[i][j]
                    final_med2_groups[i].pop(j)
                    break
                else:
                    # join with previous
                    final_med2_groups[i][j-1] += final_med2_groups[i][j]
                    final_med2_groups[i].pop(j)
                    break
    # End bind
    # print(final_med2_groups)
    # print(len(final_med2_groups))

    # for i in range(len(final_med2_groups)):
    #     print(sum(med_last_group[i]), sum(final_med2_groups[i]))
    global downhaul_grouping_mfirst
    global backhaul_grouping_mlast

    create_takhsis_model(Takhsis(
        med_first_group,
        final_mec_groups,
        downhaul_grouping_mfirst,
        med_last_group,
        final_med2_groups,
        backhaul_grouping_mlast,
        items_value,
        items_weight,
    ))

    # opt_model.write("takhsis-10.sol")

    # END OPTIMIZATION MODEL
    row = 108
    col = 68
    # # Displaying Optimal Solution for Takhsis
    global optimal_takhsis
    ws.set_column(col, col+len(items_weight), 3)
    for d in range(prob1.tot_days-prob1.stay_duration+1):
        for i in range(len(final_mec_groups[d])):
            for w in range(len(items_weight)):
                if optimal_takhsis[d, w, 1, i] > 0:
                    ws.write(row, col+w, optimal_takhsis[d, w, 1, i], tot_entry_fmt)
                else:
                    ws.write(row, col+w, "", title_fmt)
            row += 1

        if len(final_mec_groups[d]) < len(main_mec_groups[d]):
            row += 1

    for w in range(len(items_weight)):
        ws.write(107, col+w, items_weight[w], mec_entry_fmt)
        ws.write(row+4, col+w, items_weight[w], mec_entry_fmt)

    row += 5
    for d in range(prob2.tot_days-prob2.stay_duration+1):
        dd = d+prob1.tot_days-prob1.stay_duration+1
        for i in range(len(final_med2_groups[d])):
            for w in range(len(items_weight)):
                if optimal_takhsis[(dd, w, 1, i)] > 0:
                    ws.write(row, col+w,
                             optimal_takhsis[(dd, w, 1, i)], tot_entry_fmt)
                else:
                    ws.write(row, col+w, "", title_fmt)
            row += 1

        if len(final_med2_groups[d]) < len(old_final_med2_groups[d]):
            row += 1  # Shouldnt be used-for med last manually handle 1 multiples - no empty row createds

    # chart
    chart = workbook.add_chart({'type': 'column', 'subtype': 'stacked'})
    chart.set_size({'width': 2080, 'height': 700})
    chart.set_legend({'position': 'overlay_right', 'font': {'size': 20, 'bold': True}})
    chart.set_chartarea({'border': {'none': True}})
    chart.set_title({
        'name': 'Mecca Residence Plan',
        'name_font': {'size': 28, 'bold': True, 'layout': {'x': 0.01, 'y': 0.01, }}
    })
    chart.set_x_axis({
        'num_font': {'size': 18, 'bold': True}
    })
    chart.set_y_axis({
        'num_font': {'size': 18, 'bold': True}
    })
    chart.add_series({
        'name': 'Total Med First',
        'categories': ['Residence Plan', 6, 3, 6, 3+med_first_backhaul_start_day+prob2.tot_days],
        'values':     ['Residence Plan', tot_med_first_row, 3, tot_med_first_row, 3+med_first_backhaul_start_day+prob2.tot_days],
        'line':       {'color': '#4F81BD'},
        'gap': 0
    })
    chart.add_series({
        'name': 'Total Med Last',
        'categories': ['Residence Plan', 6, 3, 6, 3+med_first_backhaul_start_day+prob2.tot_days],
        'values':     ['Residence Plan', tot_med_last_row, 3, tot_med_last_row, 3+med_first_backhaul_start_day+prob2.tot_days],
        'line':       {'color': '#C0504D'},
        'gap': 0
    })

    ws.insert_chart(row+7, 10, chart)

    workbook.close()


def flight_sol_grouping(x, xres, flight_capacity, downhaul_mfirst):
    X = np.array(x)
    XRes = np.array(xres)

    dim1 = len(flight_capacity)  # f
    dim2 = len(x[0])  # d
    dim3 = len(x[0][0])  # r

    daily_pass_route = np.zeros((dim2, dim3), dtype=int)

    for d in range(dim2):
        for r in range(dim3):
            dr = 0
            for f in range(dim1):
                dr = X[f, d, r] * flight_capacity[f] - XRes[f, d, r]

            daily_pass_route[d, r] = dr

    if downhaul_mfirst:
        return daily_pass_route[:, :int(dim3/2)]
    else:
        res = daily_pass_route[:, int(dim3/2):]
        return res[np.sum(res, axis=1).nonzero()]
