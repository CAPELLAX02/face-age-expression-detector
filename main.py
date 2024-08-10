import cv2
from utils.face_detection import detect_faces
from tensorflow.keras.models import load_model

# Modelleri yükle
age_model = load_model('training/models/age_model.h5')
expression_model = load_model('training/models/expression_model.keras')

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Yüzleri algıla
    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        face_resized = face_resized / 255.0  # Normalleştirme

        # Yaş tahmini
        age_prediction = age_model.predict(face_resized.reshape(1, 64, 64, 3))

        # Yüz ifadesi tahmini
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face_resized = cv2.resize(gray_face, (48, 48))
        gray_face_resized = gray_face_resized / 255.0  # Normalleştirme
        emotion_prediction = expression_model.predict(
            gray_face_resized.reshape(1, 48, 48, 1))

        # Sonuçları ekranda göster
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Age: {int(
            age_prediction[0][0])}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        emotion_labels = ['Angry', 'Disgust', 'Fear',
                          'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_index = emotion_prediction.argmax()
        emotion_text = emotion_labels[emotion_index]
        cv2.putText(frame, f'Emotion: {
                    emotion_text}', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Face Age and Expression Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
