import math
import random
import numpy as np
import pygad


def medLine(point1, point2):
    x1 = point1[0]  # 取四點座標
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = x1**2 - x2**2 + y1**2 - y2**2
    # Ax+By+C = 0
    p0 = -C / B
    p1 = -(A + C) / B
    return [0, p0, 1, p1]


def cross_point(line1, line2):  # 計算交點函數
    x1 = line1[0]  # 取四點座標
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 計算k1,由於點均爲整數，需要進行浮點數轉化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型轉浮點型是關鍵
    if (x4 - x3) == 0:  # L2直線斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def run(contourPointsArr, population_size=1000, generations=3):
    # check if the contourPointsArr is a list
    if not isinstance(contourPointsArr[0], list):
        contourPointsArr[0] = contourPointsArr[0].tolist()
    if not isinstance(contourPointsArr[1], list):
        contourPointsArr[1] = contourPointsArr[1].tolist()

    def fitness_func(solution, solution_idx):
        output = 0
        for point in contourPointsArr[0]:
            output += math.pow(
                (solution[0] - point[0]) * (solution[0] - point[0])
                + (solution[1] - point[1]) * (solution[1] - point[1])
                - solution[4] * solution[4],
                2,
            )
        for point in contourPointsArr[1]:
            output += math.pow(
                (solution[2] - point[0]) * (solution[2] - point[0])
                + (solution[3] - point[1]) * (solution[3] - point[1])
                - solution[4] * solution[4],
                2,
            )
        output /= len(contourPointsArr[0]) + len(contourPointsArr[1])
        fitness = 1.0 / np.abs(output)
        return fitness

    leftPoints = []
    rightPoints = []
    radii = []
    for _ in range(round(math.sqrt(len(contourPointsArr[0])))):
        try:
            selectedPoints = random.sample(contourPointsArr[0], k=3)
            line1 = medLine(selectedPoints[0], selectedPoints[1])
            line2 = medLine(selectedPoints[1], selectedPoints[2])

            circleCenter = cross_point(line1, line2)
            circleCenter = np.array(circleCenter, dtype=int)

            radius = 0
            for point in selectedPoints:
                radius += math.dist(circleCenter, point)
            radius /= 3

            if circleCenter[0] <= 0 or circleCenter[1] <= 0:
                continue
            leftPoints.append([circleCenter[0], circleCenter[1]])
            radii.append(radius)
        except:
            pass
    for _ in range(round(math.sqrt(len(contourPointsArr[1])))):
        try:
            selectedPoints = random.sample(contourPointsArr[1], k=3)
            line1 = medLine(selectedPoints[0], selectedPoints[1])
            line2 = medLine(selectedPoints[1], selectedPoints[2])

            circleCenter = cross_point(line1, line2)
            circleCenter = np.array(circleCenter, dtype=int)

            radius = 0
            for point in selectedPoints:
                radius += math.dist(circleCenter, point)
            radius /= 3

            if circleCenter[0] <= 0 or circleCenter[1] <= 0:
                continue
            rightPoints.append([circleCenter[0], circleCenter[1]])
            radii.append(radius)
        except:
            pass
    initial_population = []

    for leftPoint in leftPoints:
        for rightPoint in rightPoints:
            for radius in radii:
                initial_population.append(
                    [leftPoint[0], leftPoint[1], rightPoint[0], rightPoint[1], radius]
                )
    initial_population = random.choices(
        initial_population,
        k=population_size
        if len(initial_population) > population_size
        else len(initial_population),
    )

    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=30,
        num_genes=5,
        initial_population=initial_population,
        mutation_num_genes=2,
        mutation_type="random",
        random_mutation_min_val=-10.0,
        random_mutation_max_val=10.0,
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    return solution, solution_fitness, solution_idx
