import json

data_dir = None
data_dir_09_05_23 = None
data_dir_16_05_23 = None
data_dir_13_06_23 = None
start_pulse_without_ultem = None
finish_pulse_without_ultem = None
start_pulse_after_ultem = None
finish_pulse_after_ultem = None
size_good_pulse_without_pulse_length = None
start_good_pulse = None
finish_good_pulse = None
start_vector = None
finish_vector = None
threshold_Self_MP_Matrix = None
threshold_Self_MP_Conv = None
threshold_Self_OMP = None
start_pulse_after_palbam_with_ultem_and_5000_micron = None
finish_pulse_after_palbam_with_ultem_and_5000_micron = None
size_good_pulse_air_without_pulse_length_2 = None
start_good_pulse_air = None
finish_good_pulse_air = None
start_split_filtered_pulse_500micron = None
finish_split_filtered_pulse_500micron = None
limit_omp_syntetic = None
limit_omp_real_signal = None
start_aoi = None
finish_aoi = None


# writing to json
def CreateJson():
    params = {"data_dir": '//NDTGPU/Data1/Noya/SuperResolution/data/Ultem-Air-Exp/',
              "data_dir_16_05_23": '//NDTGPU/Data1/Noya/SuperResolution/data/16 05 23/',
              "data_dir_09_05_23": '//NDTGPU/Data1/Noya/SuperResolution/data/09 05 23/',
              "data_dir_13_06_23": '//NDTGPU/Data1/Noya/SuperResolution/data/13 06 23/',
              "start_pulse_without_ultem": 2020,
              "finish_pulse_without_ultem": 2111,
              "start_pulse_after_ultem": 354,
              "finish_pulse_after_ultem": 445,
              "size_good_pulse_without_pulse_length": 2320,
              "start_good_pulse": 1060,
              "finish_good_pulse": 1170,
              "start_vector": 50,
              "finish_vector": 1101,
              "threshold_Self_MP_Matrix": 0.1215,  # 0.38?
              "threshold_Self_MP_Conv": 0.372,
              "threshold_Self_OMP": 0.38,
              "start_pulse_after_palbam_with_ultem_and_5000_micron": 2843,  # 2856  # 2823
              "finish_pulse_after_palbam_with_ultem_and_5000_micron": 3040,  # 2988  # 3020
              "size_good_pulse_air_without_pulse_length_2": 2215,
              "start_good_pulse_air": 984,  # 950
              "finish_good_pulse_air": 1145,
              "start_split_filtered_pulse_500micron": 1840,
              "finish_split_filtered_pulse_500micron": 3000,  # 3215, 3000 is good until 3000 gap, and after it cuts
              # the second peak, if we will want more data, big data, we have to make cop bigger, they need the same
              # size
              "limit_omp_syntetic": 0.1,
              "limit_omp_real_signal": 0.35,
              "start_aoi": 1980,
              "finish_aoi": 3140
              }

    with open("Params.json", "w", encoding='utf-8') as jsonParameters:
        json.dump(params, jsonParameters)


# reading from json
def ReadJson():
    with open("Params.json", "r", encoding='utf-8') as jsonParameters:
        data_loaded = json.load(jsonParameters)

    print(data_loaded)

    data_dir = data_loaded["data_dir"]
    data_dir_16_05_23 = data_loaded["data_dir_16_05_23"]
    data_dir_09_05_23 = data_loaded["data_dir_09_05_23"]
    data_dir_13_06_23 = data_loaded["data_dir_13_06_23"]
    start_pulse_without_ultem = data_loaded["start_pulse_without_ultem"]
    finish_pulse_without_ultem = data_loaded["finish_pulse_without_ultem"]
    start_pulse_after_ultem = data_loaded["start_pulse_after_ultem"]
    finish_pulse_after_ultem = data_loaded["finish_pulse_after_ultem"]
    size_good_pulse_without_pulse_length = data_loaded["size_good_pulse_without_pulse_length"]
    start_good_pulse = data_loaded["start_good_pulse"]
    finish_good_pulse = data_loaded["finish_good_pulse"]
    start_vector = data_loaded["start_vector"]
    finish_vector = data_loaded["finish_vector"]
    threshold_Self_MP_Matrix = data_loaded["threshold_Self_MP_Matrix"]
    threshold_Self_MP_Conv = data_loaded["threshold_Self_MP_Conv"]
    threshold_Self_OMP = data_loaded["threshold_Self_OMP"]
    start_pulse_after_palbam_with_ultem_and_5000_micron = data_loaded[
        "start_pulse_after_palbam_with_ultem_and_5000_micron"]
    finish_pulse_after_palbam_with_ultem_and_5000_micron = data_loaded[
        "finish_pulse_after_palbam_with_ultem_and_5000_micron"]
    size_good_pulse_air_without_pulse_length_2 = data_loaded["size_good_pulse_air_without_pulse_length_2"]
    start_good_pulse_air = data_loaded["start_good_pulse_air"]
    finish_good_pulse_air = data_loaded["finish_good_pulse_air"]
    start_split_filtered_pulse_500micron = data_loaded["start_split_filtered_pulse_500micron"]
    finish_split_filtered_pulse_500micron = data_loaded["finish_split_filtered_pulse_500micron"]
    limit_omp_syntetic = data_loaded["limit_omp_syntetic"]
    limit_omp_real_signal = data_loaded["limit_omp_real_signal"]
    start_aoi = data_loaded["start_aoi"]
    finish_aoi = data_loaded["finish_aoi"]


CreateJson()

# import json

with open("Params.json", "r", encoding='utf-8') as jsonParameters:
    data_loaded = json.load(jsonParameters)

print(data_loaded)

data_dir = data_loaded["data_dir"]
data_dir_16_05_23 = data_loaded["data_dir_16_05_23"]
data_dir_09_05_23 = data_loaded["data_dir_09_05_23"]
data_dir_13_06_23 = data_loaded["data_dir_13_06_23"]
start_pulse_without_ultem = data_loaded["start_pulse_without_ultem"]
finish_pulse_without_ultem = data_loaded["finish_pulse_without_ultem"]
start_pulse_after_ultem = data_loaded["start_pulse_after_ultem"]
finish_pulse_after_ultem = data_loaded["finish_pulse_after_ultem"]
size_good_pulse_without_pulse_length = data_loaded["size_good_pulse_without_pulse_length"]
start_good_pulse = data_loaded["start_good_pulse"]
finish_good_pulse = data_loaded["finish_good_pulse"]
start_vector = data_loaded["start_vector"]
finish_vector = data_loaded["finish_vector"]
threshold_Self_MP_Matrix = data_loaded["threshold_Self_MP_Matrix"]
threshold_Self_MP_Conv = data_loaded["threshold_Self_MP_Conv"]
threshold_Self_OMP = data_loaded["threshold_Self_OMP"]
start_pulse_after_palbam_with_ultem_and_5000_micron = data_loaded["start_pulse_after_palbam_with_ultem_and_5000_micron"]
finish_pulse_after_palbam_with_ultem_and_5000_micron = data_loaded[
    "finish_pulse_after_palbam_with_ultem_and_5000_micron"]
size_good_pulse_air_without_pulse_length_2 = data_loaded["size_good_pulse_air_without_pulse_length_2"]
start_good_pulse_air = data_loaded["start_good_pulse_air"]
finish_good_pulse_air = data_loaded["finish_good_pulse_air"]
start_split_filtered_pulse_500micron = data_loaded["start_split_filtered_pulse_500micron"]
finish_split_filtered_pulse_500micron = data_loaded["finish_split_filtered_pulse_500micron"]
limit_omp_syntetic = data_loaded["limit_omp_syntetic"]
limit_omp_real_signal = data_loaded["limit_omp_real_signal"]
start_aoi = data_loaded["start_aoi"]
finish_aoi = data_loaded["finish_aoi"]
