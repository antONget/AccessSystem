import face_recognition
from PIL import Image, ImageDraw
import pickle
import cv2
import os

def face_rec():
    zveri_face_img = face_recognition.load_image_file('images/zveri_4.jpg')
    zveri_face_img_location = face_recognition.face_locations(zveri_face_img)
    print(zveri_face_img_location)
    print(f"Found {len(zveri_face_img_location)} face(s) in this image")

    pil_img = Image.fromarray(zveri_face_img)
    draw = ImageDraw.ImageDraw(pil_img)

    for (top, right, bottom, left) in zveri_face_img_location:
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw
    pil_img.save('images/new_zveri_4.jpg')

def extacting_faces(img_path):
    count = 0
    faces = face_recognition.load_image_file(img_path)
    faces_locations = face_recognition.face_locations(faces)

    for faces_location in faces_locations:
        top, right, bottom, left = faces_location

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f'images/{count}_face_img.jpg')
        count += 1

    return f"Found {count} face(s) in this photo"

def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encoding = face_recognition.face_encodings(img1)[0]
    # print(img1_encoding)
    img2 = face_recognition.load_image_file(img2_path)
    img2_encoding = face_recognition.face_encodings(img2)[0]
    # print(img2_encoding)

    result = face_recognition.compare_faces([img1_encoding], img2_encoding)
    print(result)


def detect_person_in_video():
# функция обнаруживает и распазнает лица в видеопатоке
    # загрузка encoding-вектора известных лиц
    list_filename_encoding = os.listdir('Enoodings_faces_persons')
    data = {}
    for filename_enciding in list_filename_encoding:

        data = pickle.loads(open("Veronika _encidings.pickle", "rb").read())
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, image = video.read()

        locations = face_recognition.face_locations(image, model='hog')
        encodings = face_recognition.face_encodings(image, locations)

        for face_encoding, face_location in zip(encodings, locations):
            result = face_recognition.compare_faces(data["encodings"], face_encoding)
            match = None

            if True in result:
                match = data["name"]
                print(f'Hi {match}')
            else:
                print('Ho is')

            left_top = (face_location[3], face_location[0])
            right_bottom = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, left_top, right_bottom, color, 4)

            left_bottom = (face_location[3], face_location[2])
            right_bottom = (face_location[1], face_location[2] + 20)
            cv2.rectangle(image, left_bottom, right_bottom, color, cv2.FILLED)
            cv2.putText(
                image,
                match,
                (face_location[3] + 10, face_location[2] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                4
            )

        cv2.imshow('detect_person_in _video', image)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break

def main():
    # face_rec()
    # print(extacting_faces("images/zveri_3.jpg"))
    # compare_faces('images/roma_1.jpg', 'images/roma_2.jpg')
    detect_person_in_video()
if __name__ == '__main__':
    main()