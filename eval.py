from pdf2image import convert_from_path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import openpyxl

pages = convert_from_path("Fragebogen.pdf")
WIDTH = 250
HEIGHT = 100
page_locations_and_cells = {
    0 : [
        ((1250,1850), (3,4)), 
        ((1325,1850), (3,5)),
        ((1400,1850), (3,6))
        ],
    1 : [
        ((250,725), (3,7)), 
        ((310,725), (3,8)),
        ((360,725), (3,9)),
        ((410,725), (3,10)), 
        ((470,725), (3,11)),
        ((530,725), (3,12)),
        ((610,725), (3,13)), 
        ((670,725), (3,14)),
        ((740,725), (3,15)),
        ((810,725), (3,16)), 
        ((870,725), (3,17)),
        ((930,725), (3,18)),
        ((1050,725), (3,19)), 
        ((1110,725), (3,20)),
        ((1180,725), (3,21)),
        ((1240,725), (3,22)), 
        ((1300,725), (3,23)),
        ((280,1865), (3,24)), 
        ((340,1865), (3,25)),
        ((400,1865), (3,26)),
        ((460,1865), (3,27)), 
        ((520,1865), (3,28)),
        ((580,1865), (3,29)),
        ((640,1865), (3,30)), 
        ((700,1865), (3,31)),
        ((760,1865), (3,32)),
        ((820,1865), (3,33)), 
        ((880,1865), (3,34)),
        ((1120,1865), (3,35)),
        ((1180,1865), (3,36)), 
        ((1250,1865), (3,37)),
        ((1320,1865), (3,38)),
        ((1390,1865), (3,39))
        ]
}

workbook = openpyxl.load_workbook("wichern_selbstverantworltiches_lernen_pfingsten2022_print_leer.xlsx")
worksheet = workbook["Answers"]

def find_most_likely(img, location):    
    #Convert it to grayscale
    img_cropped = img[location[0]+20:location[0]+HEIGHT-20, location[1]+30:location[1]+WIDTH-30]
    cv2.imshow("tested", img_cropped)
    cv2.waitKey()
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.GaussianBlur(img_gray,(1,1),0)
    # thresh = 230
    # img_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    all_vals = []
    for i in range(1, 7):
        # Read the template
        template = cv2.imread(f'{i}.png', 0)
        # thresh = 150
        # template_bw = cv2.threshold(template, thresh, 255, cv2.THRESH_BINARY)[1]
        
        # Store width and height of template in w and h
        # w, h = template.shape[::-1]
        
        # Perform match operations.
        res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)

        # Specify a threshold
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # Store the coordinates of matched area in a numpy array
        # loc = np.unravel_index(res.argmax(), res.shape)
        # loc = max_loc
        all_vals.append(min_val)

        # Draw a rectangle around the matched region.
        # for pt in zip(*loc[::-1]):
        #     print(pt)
        # detected = img_gray.copy()
        # cv2.rectangle(detected, loc, (loc[0] + w, loc[1] + h), (0, 255, 255), 2)
        
        # # # # Show the final image with the matched area.
        # cv2.imshow('Detected', detected)
        # cv2.waitKey()
    best_val = min(all_vals)
    best = all_vals.index(best_val) + 1
    # all_vals[best - 1] = -1
    # second_best_val = max(all_vals)
    # print(best_val, best_val - second_best_val, all_vals.index(second_best_val) + 1)
    return best

for p in page_locations_and_cells.keys():
    img = np.array(pages[p])
    img_half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Half Image', img_half)
    cv2.waitKey()
    for l,c in page_locations_and_cells[p]: 
        match = find_most_likely(img, l)
        print(f"Most likely match: {match}")
        worksheet.cell(row = c[0], column=c[1]).value = match

workbook.save("changed.xlsx")

    # blur = cv2.pyrMeanShiftFiltering(img, 11, 21)
    # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    #     if len(approx) == 4:
    #         x,y,w,h = cv2.boundingRect(approx)
    #         cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,12),2)

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('image', img)
    # cv2.waitKey()

    # img1 =  cv2.GaussianBlur(img,(5,5),0)
    # img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    # f1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    # f1 = 255 - cv2.threshold(f1, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # fgdilated = cv2.dilate(f1, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) , iterations = 1)
    # fgclosing = cv2.morphologyEx(fgdilated, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)))

    # plt.imshow(fgclosing)
    # plt.show()


# img2, contours, hierarchy = cv2.findContours(fgdilated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
#     #print(cnt)
#     #print(cv2.contourArea(cnt))
#     if cv2.contourArea(cnt) > 200:
#         #hull = cv2.convexHull(cnt)
#         #print(hull)
#         #cv2.drawContours(img, [hull], -1, (255, 255, 255), 1)

#         (x,y,w,h) = cv2.boundingRect(cnt)
#         if h >7:
#             cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 1)

# plt.imshow(img)
# plt.show()