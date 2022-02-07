import cv2
from protopost import protopost_client as ppcl
from nd_to_json import nd_to_json, json_to_nd

STEP = lambda: ppcl("http://127.0.0.1:8080")
ADD_POTENTIAL = lambda x: ppcl("http://127.0.0.1:8080/add-potential", x)
GET_SPIKES = lambda: ppcl("http://127.0.0.1:8080/get-spikes")
REWARD = lambda r: ppcl("http://127.0.0.1:8080/reward", r)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)

while True:
  STEP()
  spikes = GET_SPIKES()
  spikes = json_to_nd(spikes)
  REWARD(1)
  cv2.imshow("img", spikes)
  cv2.waitKey(1)
