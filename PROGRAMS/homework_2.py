import numpy as np
from calibration_library import *
from dataParsing_library import *
from validation_test import *
import copy
import os
import re

def main(): 
    # Read in input dataset
    script_directory = os.path.dirname(__file__)
    dirname = os.path.dirname(script_directory)
    choose_set = 'a'
    #base_path = os.path.join(dirname, 'pa2_student_data\\PA2 Student Data\\pa2-unknown-')
    base_path = os.path.join(dirname, 'pa2_student_data\\PA2 Student Data\\pa2-debug-')

    calbody = base_path + choose_set + '-calbody.txt'
    calbody_point_cloud = parseData(calbody)
    d0, a0, c0 = parseCalbody(calbody_point_cloud)

    calreading = base_path + choose_set + '-calreadings.txt'
    calreading_point_cloud = parseData(calreading)
    calreading_frames = parseFrame(calreading_point_cloud, 8+8+27) # 8 optical markers on calibration object and 27 EM markers on calibration object
    
    # new inputs to be incorporated 
    ct_fid = base_path + choose_set + '-ct-fiducials.txt'
    em_fid = base_path + choose_set + '-em-fiducialss.txt'
    em_nav = base_path + choose_set + '-EM-nav.txt'

    empivot = base_path + choose_set + '-empivot.txt'
    empivot_point_cloud = parseData(empivot)
    empivot_frames = parseFrame(empivot_point_cloud, 6) # stores the list of 12 frames, each of which contains data of 6 EM markers on probe 
    
    optpivot = base_path + choose_set + '-optpivot.txt'
    optpivot_point_cloud = parseData(optpivot) # stores the list of 12 frames, each of which contains data of 8 optical markers on EM base & 6 EM markers on probe
    optpivot_em_frames, optpivot_opt_frames = parseOptpivot(optpivot_point_cloud, 8, 6) 
    

    registration = setRegistration()
    np.set_printoptions(formatter={'float': '{:.2f}'.format})

    # Q4
    # Part A 
    source_points_d = d0
    trans_matrix_d = []
    target_points = []

    for i in range(8):
        target_points = calreading_frames[i][:8]
        transformation_matrix = registration.calculate_3d_transformation(source_points_d, target_points)
        trans_matrix_d.append(transformation_matrix)

    # Part B
    source_points_a = a0
    trans_matrix_a = []
    target_points = []

    for i in range(8):
        target_points = calreading_frames[i][8:16]
        transformation_matrix = registration.calculate_3d_transformation(source_points_a, target_points)
        trans_matrix_a.append(transformation_matrix)
    
    # Part C
    source_points_c = c0
    transformation_matrix = []
    transformed_point = []
    for i in range(8):
        transformation_matrix = np.dot(np.linalg.inv(trans_matrix_d[i]), trans_matrix_a[i])
        transformed_point.append(registration.apply_transformation(source_points_c, transformation_matrix))
    
    # Part D
    # print(transformed_point)

    # Q5
    # Initalize the set for gj = Gj - G0
    translated_points_Gj = copy.deepcopy(empivot_frames)
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    midpoint = np.mean(empivot_frames, axis=1)
    trans_matrix_FG = []

    for i in range(12):
        for j in range(6):
            p = empivot_frames[i][j] - midpoint[i]
            translated_points_Gj[i][j] = p
    
    # fix gj as the original starting positions
    source_points = translated_points_Gj[0]
    for i in range(12):
        target_points = empivot_frames[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points, target_points)
        trans_matrix_FG.append(transformation_matrix)
    p_tip_G, p_pivot_G = registration.pivot_calibration(trans_matrix_FG)

    # Q6
    # Initalize the set for hj = Hj - H0
    translated_points = copy.deepcopy(optpivot_opt_frames)
    H_prime = []
    # Find centroid of Hj (the original position of 6 EM markers on the probe)
    midpoint = np.mean(optpivot_opt_frames, axis=1)
    # Find transformation matrix for 12 frames
    trans_matrix_f = []

    # Calculate Fd
    source_points_d = d0 #optpivot_em_frames[0]
    transformation_matrix_Fd = []
    target_points = []

    for i in range(12):
        target_points = optpivot_em_frames[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points_d, target_points)
        transformation_matrix = np.linalg.inv(transformation_matrix)
        transformation_matrix_Fd.append(transformation_matrix)

    for i in range(12):
        for j in range(6):
        # fill out hj 
            p = optpivot_opt_frames[i][j] - midpoint[i]
            translated_points[i][j] = p
    
    #apply Fd to H
    for i in range(12):
        chunk_array = np.vstack(optpivot_opt_frames[i])
        transformed_chunk = registration.apply_transformation(chunk_array, transformation_matrix_Fd[i])
        H_prime.append(transformed_chunk)

    # fix Hj as the original starting positions
    source_points = translated_points[0]

    for i in range(12):
        target_points = H_prime[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points, target_points)
        trans_matrix_f.append(transformation_matrix)
    p_tip_H, p_pivot_H = registration.pivot_calibration(trans_matrix_f)

    # format output
    # Output 1
    output_name_cal = 'pa2-unknown-' + choose_set + '-output1.txt'
    input = str(p_pivot_G) + str(p_pivot_H) + str(transformed_point)
    lines = input.strip().split("\n")
    converted_lines = []

    for line in lines:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        groups_of_three = [numbers[i:i + 3] for i in range(0, len(numbers), 3)]
        formatted_groups = [", ".join([f"{float(num):7.2f}" for num in group]) for group in groups_of_three]
        converted_lines.extend(formatted_groups)
    output_string = "\n".join(converted_lines)


    # Output 2
    output_name_ct = 'pa2-unknown-' + choose_set + '-output2.txt'

    # write to output
    with open(output_name_cal, "w") as file:
        file.write('27, 8, ' + output_name_cal + '\n')
        file.write(output_string)

if __name__ == "__main__":
    main()
    '''
    v = validate()
    file1 = 'pa1_student_data\PA1 Student Data\pa1-debug-g-output1.txt'
    file2 = 'OUTPUT\pa1-unknown-g-output.txt'
    
    percentage_differences = v.calculate_error_from_sample(file1, file2, use_reference=0)
    print(np.mean(percentage_differences))
    '''