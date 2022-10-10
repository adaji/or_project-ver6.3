import copy


def build_mec_groups(g1, g2):
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

    print('Main Groups')
    for i in range(len(new_g)):
        print(new_g[i])
    main_groups = copy.deepcopy(new_g)
    print("********")

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

    print('Final groups binding small groups to larger ones')
    for i in range(len(new_g)):
        print(new_g[i])

    print("Build_MEC Output")
    print("Main Groups")
    print(main_groups)
    print("New G")
    print(new_g)
    return main_groups, new_g


g1 = [2097, 2097, 2097, 2097, 2097, 2097, 2097, 2097, 2097, 2097, 2097, 2089, 2097,
      2097, 1864, 2097, 2097, 2097, 1864, 1864, 1864, 1864, 1864, 1864, 1864, 1864]

g2 = [4427, 4427, 4427, 4427, 2563, 2563, 2563, 2563, 2563,
      2563, 2563, 2330, 2563, 2330, 2330, 2563, 2330, 2322]
