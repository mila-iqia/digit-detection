import json

#  dataset_dir = '/home/jerpint/digit-detection/data/Avenue/Humanware_v1_1551895483/test'
#  text_file = '/home/jerpint/digit-detection/data/Avenue/Humanware_v1_1551895483/test_files.txt'
#  bbox_file = '/home/jerpint/digit-detection/inference/avenue_test_set/bbox.json'
#  json_file = '/home/jerpint/digit-detection/data/Avenue/Humanware_v1_1551895483/test/instances_test.json'


json_file = '/home/jerpint/digit-detection/data/Avenue/Humanware_v1_1553272293/test_sample/instances_test.json'
json_outfile = '/home/jerpint/digit-detection/data/Avenue/Humanware_v1_1553272293/test_sample/instances_test_sample.json'

#  with open(text_file, 'r') as f:
#      files = f.read().splitlines()

#  with open(bbox_file) as f:
#      bbox = json.load(f)

with open(json_file) as f:
    metadata = json.load(f)


sample = {}

sample = metadata.copy()

keys = ['images', 'annotations']

for key in keys:
    sample[key] = metadata[key][0:100]


with open(json_outfile, 'w') as outfile:
    json.dump(sample, outfile)
