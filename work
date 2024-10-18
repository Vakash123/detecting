import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.utils import long_to_uint64
from common.FPS import PERF_DATA
import numpy as np
import pyds
import cv2
import os
import os.path
from os import path
import json
from datetime import datetime
import psycopg2
import regex as re
import requests
import pytz
from pytz import timezone
import base64
import time
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import torch
from ultralytics YOLOimport

perf_data = None

MAX_DISPLAY_LEN = 64
MAX_TIME_STAMP_LEN = 32

MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 1920F
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
file_loop = True
LOITERING_THRESHOLD = 300

global PGIE_CLASS_ID_PERSON

PGIE_CLASS_ID_PERSON = 0


pgie_classes_str = ["person"]
global saved_track_ids
saved_track_ids = set()

BASE_DIR = os.path.realpath(__file__)
config_path = 'config.json'
with open(config_path) as config_buffer:
    static_config = json.load(config_buffer)

class Solution:
    def __init__(self, config):
        config = static_config
        self.model_id = 76
        self.camera_name = "ANPR"
        self.model_name = "ANPR"
        self.camera_id = 1
        self.commisionerate = "commisionerate"
        self.zone = "zone"
        self.police_station = "police_station"
        self.junction_name = "junction"
        self.line_coordinate_points = [1, 446, 1901, 586]  # Define a sample line (not used now)
        self.input_path = "/home/shireesha/Documents/Person_Loitering/Loitering/input/Raithu_bazar-bustop_HumayunagarPS1.mp4"
 
        self.model_path = config['model']['model_path']
        self.model = YOLO(SRC_DIR + self.model_path)
        self.person_centroids = {}
        self.MAX_DURATION = 30  # Max duration for loitering in seconds
 
        self.ROI = (100, 100, 1200, 800)  # Define ROI as (x_min, y_min, x_max, y_max)
        self.is_benchmark = False
        self.main()



class Yolov8(object):
    def __init__(self, config):
        # get params from config file
        
        config = static_config
        self.model_id = config['model']['model_id']
        self.PGIE_CONFIG_FILE = config['model']['PGIE_CONFIG_FILE']
        self.tracker_config_file=config['model']['tracker_config_file']

        
        self.app_api_url = config['application']['api_url']
        self.image_server_api_url = config['image_server']['api_url']

        self.alert_message = config['model']['alert_message']
        self.backend_api_url = config['backend']['api_url']
        self.last_alert_time = {}
        self.sample_data = self.read_db_to_check_for_cameras_added()

        for entry in self.sample_data['data']:
            entry['model_name'] = self.transform_string(entry['model_name'])
            #entry['camera_name'] = self.transform_string(entry['camera_name'])
            entry['model_id'] = self.model_id
            # self.alert_interval = int(entry['alert_time(sec)'])
            self.analytic_type = entry['analytic_type']
            alert_time = entry['alert_time(sec)']
            if alert_time is None or alert_time == '' or alert_time =='null':
                self.alert_interval = 30
            else:
                self.alert_interval = int(alert_time)

        print("sample_Data........", self.sample_data)
        # self.main()

    def transform_string(self, input_str):
        return re.sub(r'\W', ' ', input_str).replace(' ', '-').replace('_', '-')

    def read_db_to_check_for_cameras_added(self):
        try:
            print("model id is ", self.model_id)
            if self.model_id:
                res = requests.get(url=f"{self.backend_api_url}?model_id=" + self.model_id, headers={
                    'content-type': 'application/json'})

                response_json = res.json()
                for idx, item in enumerate(response_json.get('data', [])):
                    item['id'] = idx
                return response_json

        except Exception as e:
            print(e)

    def sendAlert(self, frame, alert_data) -> None:
        date_time_str = datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f')
        date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
        saving_filename = alert_data['model_name'] + "_" + str(alert_data['camera_id']) + "_" + str(date_time_obj) + ".jpg"
        sent_alert = {}
        cv2.putText(frame, "Loitering Detected", (25, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        retval, buffer = cv2.imencode('.jpg', frame)

        sent_alert['image'] = base64.b64encode(buffer).decode()
        sent_alert['time'] = str(date_time_str)
        sent_alert['model_name'] = alert_data['model_name']
        sent_alert['camera_id'] = alert_data['camera_id']

        r = requests.post(url=self.image_server_api_url, headers={'content-type': 'application/json'},
                          data=json.dumps(sent_alert))
        print("Response from Image Server API", r)

        alert_data1 = {}
        alert_data1['info'] = self.alert_message
        alert_data1['model_name'] = alert_data['model_name']
        alert_data1['model_id'] = alert_data['model_id']
        alert_data1['camera_name'] = alert_data['camera_name']
        alert_data1['junction_name'] = alert_data['junction_name']
        alert_data1['camera_id'] = alert_data['camera_id']
        alert_data1['time'] = str(date_time_obj)
        alert_data1['commisionerate'] = alert_data['commisionerate']
        alert_data1['zone'] = alert_data['zone']
        alert_data1['police_station'] = alert_data['police_station']
        alert_data1['path'] = saving_filename
        alert_data1['analytic_type']=self.analytic_type

        sent_alert = {}
        sent_alert['data'] = alert_data1
        r = requests.post(url=self.app_api_url, headers={'content-type': 'application/json'},
		 					data=json.dumps(sent_alert))
        print("Response from Alert API", r)


    def generate_vehicle_meta(self,data,alert_data,pad_index):
        obj = pyds.NvDsVehicleObject.cast(data)
        obj.type = str(alert_data) + " " +str (pad_index)
        return obj
    
    def generate_event_msg_meta(self,data, class_id, alert_data,pad_index):
        meta = pyds.NvDsEventMsgMeta.cast(data)
        meta.ts = pyds.alloc_buffer(MAX_TIME_STAMP_LEN + 1)
        pyds.generate_ts_rfc3339(meta.ts, MAX_TIME_STAMP_LEN)

        meta.objClassId = class_id
        obj = pyds.alloc_nvds_vehicle_object()
        obj = self.generate_vehicle_meta(obj,alert_data,pad_index)
        meta.extMsg = obj
        meta.extMsgSize = sys.getsizeof(pyds.NvDsVehicleObject)
        return meta
        
    def is_point_inside_roi(self, point,roi_points):
        roi_coordinate_points = [coord for point in roi_points for coord in point] + roi_points[0]
        polygon_points = [(roi_coordinate_points[i], roi_coordinate_points[i+1]) for i in range(0, len(roi_coordinate_points), 2)]
        polygon = Polygon(polygon_points)
        point = Point(point)
        return polygon.contains(point)
        
    def xyxy_to_xy(self, x1, y1, x2, y2):
        try:
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            return x, y
        except Exception as e:
            print(e)


    def tiler_sink_pad_buffer_probe(self, pad, info, u_data):
        alert_data = ''
        frame_number = 0
        num_rects = 0

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            bounding_boxes = []

            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                br
            frame_number = frame_meta.frame_num
            l_obj = frame_meta.obj_meta_list
            num_rects = frame_meta.num_obj_meta

            obj_counter = {
                PGIE_CLASS_ID_PERSON: 0
                
            }

            l_usr = frame_meta.frame_user_meta_list
            while l_usr is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_usr.data)
                except StopIteration:
                    continue

                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_USER_META:
                    custom_msg_meta = pyds.CustomDataStruct.cast(user_meta.user_meta_data)
                    Gst.info(f'event msg meta, otherAttrs = {pyds.get_string(custom_msg_meta.message)}')
                    custom_data = pyds.get_string(custom_msg_meta.message)
                    data = json.loads(custom_data)

                    for item in data['data']:
                        if item['id'] == frame_meta.pad_index:
                            alert_data = item

                try:
                    l_usr = l_usr.next
                except StopIteration:
                    break

            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                bbox_info = {
                    'top': int(obj_meta.rect_params.top),
                    'left': int(obj_meta.rect_params.left),
                    'width': int(obj_meta.rect_params.width),
                    'height': int(obj_meta.rect_params.height)
                }

                confidence = obj_meta.confidence
                class_id = obj_meta.class_id
                track_id = obj_meta.object_id
                obj_counter[obj_meta.class_id] += 1

                if confidence > 0.55 and class_id == 0:
                    x_min = bbox_info['left']
                    y_min = bbox_info['top']
                    x_max = x_min + bbox_info['width']
                    y_max = y_min + bbox_info['height']
                    current_bbox_centroid = self.xyxy_to_xy(x_min, y_min, x_max, y_max)
                    roi_points = alert_data['roi_points']
                    if roi_points:
                        if self.is_point_inside_roi(current_bbox_centroid,roi_points):
                            bounding_boxes.append(bbox_info)
                            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                            frame_copy = np.array(n_frame, copy=True, order='C')
                            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                    else:
                        bounding_boxes.append(bbox_info)
                        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                        frame_copy = np.array(n_frame, copy=True, order='C')
                        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)


                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            print(f"Frame Number={frame_number}, Number of Objects={num_rects}, Loitering detected=",obj_counter[PGIE_CLASS_ID_PERSON])
            current_time = time.time()
            pad_index = frame_meta.pad_index  # Use pad_index instead of camera_name
            # print(f"Pad Index: {pad_index}")

            if pad_index not in self.last_alert_time:
                self.last_alert_time[pad_index] = 0
                # print(f"Initialized last alert time for pad_index: {pad_index}")
            if len(bounding_boxes) > 0:
                if self.last_alert_time[pad_index] is None or (current_time - self.last_alert_time[pad_index] >= self.alert_interval):
                    self.last_alert_time[pad_index] = current_time
                    user_event_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
                    if user_event_meta:
                        msg_meta = pyds.alloc_nvds_event_msg_meta(user_event_meta)
                        msg_meta = self.generate_event_msg_meta(msg_meta, obj_meta.class_id, alert_data,
                                                                frame_meta.pad_index)

                        user_event_meta.user_meta_data = msg_meta
                        user_event_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_EVENT_MSG_META
                        pyds.nvds_add_user_meta_to_frame(frame_meta, user_event_meta)
                        for bbox in bounding_boxes:
                            top, left, width, height = bbox['top'], bbox['left'], bbox['width'], bbox['height']
                            cv2.rectangle(frame_copy, (left, top), (left + width, top + height), (0, 0, 255), 2)
                        if alert_data['roi_points'] is not None:
                            roi_points = np.array(alert_data['roi_points'], dtype=np.int32)
                            roi_points = roi_points.reshape((-1, 1, 2))
                            # print(f"polygon roi points are: {roi_points}")
                            cv2.polylines(frame_copy,[roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
                            self.sendAlert(frame_copy, alert_data)
                            # print(f"Alert sent for pad_index: {pad_index}")
                        else:
                            self.sendAlert(frame_copy, alert_data)
                            # print(f"Alert sent for pad_index: {pad_index}")

                    else:
                        print("Error in attaching event meta to buffer\n")

            global perf_data
            stream_index = "stream{0}".format(frame_meta.pad_index)
            perf_data.update_fps(stream_index)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK



    def cb_newpad(self, decodebin, decoder_src_pad, data):
        print("In cb_newpad\n")
        caps = decoder_src_pad.get_current_caps()
        if not caps:
            caps = decoder_src_pad.query_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        print("gstname=", gstname)
        if gstname.find("video") != -1:
            if features.contains("memory:NVMM"):
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    sys.stderr.write(
                        "Failed to link decoder src pad to source bin ghost pad\n"
                    )
            else:
                sys.stderr.write(
                    "Error: Decodebin did not pick nvidia decoder plugin.\n")

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        print("Decodebin child added:", name, "\n")
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.decodebin_child_added, user_data)

        if not is_aarch64() and name.find("nvv4l2decoder") != -1:
            Object.set_property("cudadec-memtype", 2)

        if "source" in name:
            source_element = child_proxy.get_by_name("source")
            if source_element.find_property('drop-on-latency') != None:
                Object.set_property("drop-on-latency", True)

    def streammux_src_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            Gst.warning("Unable to get GstBuffer ")
            return

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        pyds.nvds_acquire_meta_lock(batch_meta)
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                frame_number = frame_meta.frame_num
            except StopIteration:
                continue

            user_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
            sample_data = self.sample_data
            json_string = json.dumps(sample_data, indent=4)

            if user_meta:
                test_string = json_string
                data = pyds.alloc_custom_struct(user_meta)
                data.message = test_string
                data.message = pyds.get_string(data.message)
                data.structId = frame_number
                data.sampleInt = frame_number + 1

                user_meta.user_meta_data = data
                user_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_USER_META

                pyds.nvds_add_user_meta_to_frame(frame_meta, user_meta)
            else:
                print('failed to acquire user meta')

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        pyds.nvds_release_meta_lock(batch_meta)
        return Gst.PadProbeReturn.OK

    def create_source_bin(self, index, uri):
        print("Creating source bin")

        bin_name = "source-bin-%02d" % index
        print(bin_name)
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")

        if file_loop:
            uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
            uri_decode_bin.set_property("file-loop", 1)
            uri_decode_bin.set_property("cudadec-memtype", 0)
        else:
            uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.set_property("rtsp-reconnect-interval", 30)
        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)

        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(
            Gst.GhostPad.new_no_target(
                "src", Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin
    def calculate_centroid(self, x_min, y_min, x_max, y_max):
        x_center = (x_min + x_max) / 2
        y_center = y_max  # The bottom of the bounding box
        return x_center, y_center
    def is_inside_roi(self, centroid):
        x, y = centroid
        x_min, y_min, x_max, y_max = self.ROI
        return x_min <= x <= x_max and y_min <= y <= y_max
    def draw_roi(self, frame):
        x_min, y_min, x_max, y_max = self.ROI
        return cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # Yellow box for ROI
    def main(self) -> None:
        try:
            with torch.no_grad():
                print("Camera: " + self.camera_name + " Solution Start Time: ", datetime.now())
                frame_counter = 0
 
                vid = cv2.VideoCapture(self.input_path)
                while True:
                    hasFrame, frame = vid.read()
                    if not hasFrame:
                        print("[ERR]: no frame")
                        vid.release()
                        vid = cv2.VideoCapture(self.input_path)
                        print("[INFO]: reinitialize frame")
                        break
                    else:
                        frame_counter += 1
                        date_time = datetime.now()
                        frame = self.draw_roi(frame)  # Draw ROI on the frame
 
                        for result in self.model.track(
                                source=frame,
                                show=False,
                                persist=True,
                                stream=True,
                                agnostic_nms=False,
                                verbose=False,
                                tracker="bytetrack.yaml",
                                conf=0.6
                        ):
                            output_frame = result.orig_img
                            if output_frame is not None:
                                prediction = result.boxes.data.cpu().numpy()
                                if prediction.any():
                                    for cur_prediction in prediction:
                                        if len(cur_prediction) == 7 and (cur_prediction[6] == 0):  # Class 0 = 'person'
                                            x_min, y_min, x_max, y_max, personID, conf, cls = cur_prediction
 
                                            current_time = time.time()
                                            centroid = self.calculate_centroid(x_min, y_min, x_max, y_max)
 
                                            if not self.is_inside_roi(centroid):
                                                continue  # Skip if not inside ROI
 
                                            if personID not in self.person_centroids:
                                                # New person, initialize tracking data
                                                self.person_centroids[personID] = {'initial_time': current_time, 'last_update_time': current_time,
                                                                                   'centroids': [centroid], 'alert_time': None}
                                            else:
                                                person_data = self.person_centroids[personID]
                                                last_update_time = person_data['last_update_time']
                                                duration_since_last_update = current_time - last_update_time
 
                                                if int(duration_since_last_update) >= 1:
                                                    # Update centroid and timestamp
                                                    self.person_centroids[personID]['centroids'].append(centroid)
                                                    self.person_centroids[personID]['last_update_time'] = current_time
 
                                                # Calculate total duration of tracking
                                                initial_time = person_data['initial_time']
                                                duration = current_time - initial_time
 
                                                if duration > self.MAX_DURATION:
                                                    alert_time = person_data['alert_time']
                                                    if alert_time is None or (current_time - alert_time) >= 30:
                                                        print(f"Loitering detected: Person {personID} in ROI for more than 30 seconds.")
 
                                                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # Green box
 
                                                        centroids = person_data['centroids']
                                                        for i in range(len(centroids) - 1):
                                                            pt1 = (int(centroids[i][0]), int(centroids[i][1]))
                                                            pt2 = (int(centroids[i + 1][0]), int(centroids[i + 1][1]))
                                                            num_dots = 10  # Number of dots for trail
                                                            for j in range(num_dots + 1):
                                                                x = int(pt1[0] + (pt2[0] - pt1[0]) * j / num_dots)
                                                                y = int(pt1[1] + (pt2[1] - pt1[1]) * j / num_dots)
                                                                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Blue dotted trail
 
                                                        # Draw latest centroid as a circle
                                                        for c in centroids:
                                                            cv2.circle(frame, (int(c[0]), int(c[1])), 5, (255, 0, 0), -1)
 
                                                        # Save image with alert
                                                        image_filename = f"alert_person_{personID}_{int(current_time)}.jpg"
                                                        cv2.imwrite(image_filename, frame)
 
                                                        self.person_centroids[personID]['alert_time'] = current_time
 
        except Exception as e:
            print(e)

    
        
if __name__ == '__main__':
    solution = Solution(static_config)
	# Create an instance of the Yolov8 class
##	yolov8_instance = Yolov8(config=static_config)
##	sys.exit(yolov8_instance.main()) 
