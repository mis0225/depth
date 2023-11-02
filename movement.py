import cv2

cap = cv2.VideoCapture

before = None
while True:
    #OpenCVでカメラの画像を読み込む
    ret, frame = cap.read()

    # スクリーンショットをとるため、サイズを1/4に縮小
    frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/74)))

    # 加工なしの画像を表示する
    cv2.imshow('Raw Frame', frame)

    # 取り込んだフレームに対して差分をとって動いているところが明るい画像を作る
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if before is None:
        before = gray.copy().astype('float')
        continue
    
    # 現フレームと前フレームの加重平均をとる
    cv2.accumulateWrighted(gray, before, 0.5)
    mdframe = cv2.absdiff(gray, cv2.convertScaleAbs(before))

    # 動いている部分を明るく表示
    cv2.imshow('Motion Detected Frame', mdframe)

    # 動いているエリアの面積を計算して検出結果を抽出
    thresh = cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]

    # 輪郭データに変換するfindContours
    image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    target = contours[0]
    for cnt in contours:
        # 輪郭の面積を求めるcontourArea
        area = cv2.contourArea(cnt)
        if max_area < area and area < 10000 and area >1000:
            max_area = area
            target = cnt

        # 動いているエリアのうち大きいものを囲んで可視化する
        if max_area <= 1000:
            areaframe = frame
            cv2.putText(areaframe, 'not detected', (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0),3, cv2.LINE_AA)
        
        else:
            # bounding box
            x,y,w,h = cv2.boundingRect(target)
            areaframe = cv2.rectangle(frame, (x, y), (x+y, y+h), (0, 255, 0), 2)

        cv2.imshow('Motion Detected Area Frame', areaframe)
        # wait 1ms for key input and for esc, break
        k = cv2.waitKey(1)
        if k == 27:
            break
        
    
    # release capture and close all windows
    cap.release()
    cv2.destroyAllWindows()