import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from PIL import ImageDraw as ImageDraw

# This is how you authenticate.


class ClarifaiPredictor:

  def __init__(self, token):
    self.stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())
    self.metadata = (('x-clarifai-session-token', token),)

  def predict_from_image(self, img_bytes, user_id, app_id, model_id, version_id):
    '''
        '''
    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
        # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
        model_id=model_id,
        version_id=version_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(base64=img_bytes)))
        ])
    response = self.stub.PostModelOutputs(request, metadata=self.metadata)
    # print(response)
    if response.status.code != status_code_pb2.SUCCESS:
      print("Request failed, status code: " + str(response.status.code))

    return response


def draw_bounding_boxes(image_draw, height, width, top_row, left_col, bottom_row, right_col):
  '''
    '''
  top = int(top_row * height)
  left = int(left_col * width)
  bottom = int(bottom_row * height)
  right = int(right_col * width)

  image_draw.line(
      [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
      width=3,
      fill='red')
  # return image_draw


# function from: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.
  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.
  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.
  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height,
                                  ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  if thickness > 0:
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin), display_str, fill='black', font=font)
    text_bottom -= text_height - 2 * margin


if __name__ == '__main__':
  API_APP_KEY = ''
  MODEL_ID = ''
  VERSION_ID = ''
  predictor = ClarifaiPredictor(API_APP_KEY)
  filep = '/Users/andrewmendez/Documents/dog_test.jpg'
  img_b = open(filep, "rb").read()
  response = predictor.predict_from_image(img_b, MODEL_ID, VERSION_ID)

  API_APP_KEY = ''
  MODEL_ID = ''
  VERSION_ID = ''
  # Object Predictor
  filep = '/Users/andrewmendez/Documents/airport.jpg'
  img_b = open(filep, "rb").read()
  response = predictor.predict_from_image(img_b, MODEL_ID, VERSION_ID)

  for region in response.outputs[0].data.regions:

    print("ID: {}, BBOX: ({}), CLASS_NAME: {}, CONFIDENCE: {}".format(
        region.id, region.region_info.bounding_box, region.data.concepts[0].name,
        region.data.concepts[0].value))

    print("*" * 10)
