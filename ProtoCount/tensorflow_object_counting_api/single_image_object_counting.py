#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Object detection imports
from tensorflow_object_counting_api.utils import backbone
from tensorflow_object_counting_api.api import object_counting_api

class SingleImageObjCount():
    def CountObjects(img):
        input = img
        detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')
        is_color_recognition_enabled = False
        result = object_counting_api.single_image_object_counting(input, detection_graph, category_index, is_color_recognition_enabled) # targeted objects counting

        print (result)
