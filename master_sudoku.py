# ----------------DEVELOPER-----------------
# *****************************************
# *****************************************
# This was originally developed by Rochan
# *****************************************
# *****************************************
# ------------------------------------------


######################## SUDOKU SOLVER ########################


# To display our message, we use tkinter
import tkinter.messagebox
# To display at the starting.
# Pygame may take time to initialize
root = tkinter.Tk().withdraw()
tkinter.messagebox.showinfo("Starting", "Please wait as we get everything ready...")


# ----------IMPORTS----------

# We require cv2 for image processing
import cv2
# We require numpy for analysis of images (conveeting into matrices)
import numpy as np
# We require tesseract for OCR
import pytesseract
# Inbuilt module - time for making waiting (delay)
import time
# Pygame will be used for making GUI for our application
import pygame
# This is a neural net for character recognition based on Tensorflow's MNIST Dataset
# import predict_2


# ----------INITIALISATIONS----------

# First we start by giving the path of the tesseract OCR
# If you download the whole folder, use this as I have already put Tesseract in this folder
# This tesseract is configured for Windows, 64 bit
# If you download only the source code, you need to specify the path of pytesseract in your system
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files (x86)/Tesseract-OCR/tesseract"

# First thing we need to do is to initialise Pygame
# This will be used for creating the basic startup GUI
pygame.init()
# Similarly, we initialise the fonts used
pygame.font.init()

# This is for creating an object for the pygame.display
# We need to initialise the font first
numberFont = pygame.font.SysFont("comicsansms", 45)

# These are some standard measurements I have used for making the grid
# I have used two basic variables
# I will be using these as the basis for all window measurements
windowMultiplier = 5
windowSize = 81

# Similarly, I have initialised the variables for the height and width of the main game window
# I have done this in terms of the windowMultiplier and windowSize
windowHeight = int(windowSize * windowMultiplier)
windowWidth = int(windowSize * windowMultiplier)

# This is the size of each square in the Sudoku Grid
squareSize = int((windowSize * windowMultiplier) / 3)

# Each cell has 9 numbers
# This is the size of each of the cell
cellSize = int(squareSize / 3)

# Inside each cell, there should be numbers
# I have taken this as the size of each number
numberSize = int(cellSize / 3)

# Now, I have initialised some of the RGB values for the colors I have used in the GUI
# RGB value of white
white = (255, 255, 255)
# RGB value of dull white or slight gray
dull_white = (207, 209, 211)
# RGB value of black
black = (0, 0, 0)
# RGB value of normal red (dull red)
red = (150, 0, 0)
# RGB value of normal green (dull green)
green = (0, 150, 0)
# RGB value of bright red
bright_red = (255, 0, 0)
# RGB value of bright green
bright_green = (0, 255, 0)


# ----------SUDOKU SOLVING ALGORITHM----------


# Credits - Geeksforgeeks
# A Backtracking program  in Python to solve Sudoku problem


# Searches the grid to find an entry that is still unassigned. If
# found, the reference parameters row, col will be set the location
# that is unassigned, and true is returned. If no unassigned entries
# remain, false is returned.
# 'l' is a list  variable that has been passed from the solve_sudoku function
# to keep track of incrementation of Rows and Columns
def find_empty_location(arr, l):
    for row in range(9):
        for col in range(9):
            if arr[row][col] == 0:
                l[0] = row
                l[1] = col
                return True
    return False


# Returns a boolean which indicates whether any assigned entry
# in the specified row matches the given number.
def used_in_row(arr, row, num):
    for i in range(9):
        if arr[row][i] == num:
            return True
    return False


# Returns a boolean which indicates whether any assigned entry
# in the specified column matches the given number.
def used_in_col(arr, col, num):
    for i in range(9):
        if arr[i][col] == num:
            return True
    return False


# Returns a boolean which indicates whether any assigned entry
# within the specified 3x3 box matches the given number
def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if arr[i + row][j + col] == num:
                return True
    return False


# Checks whether it will be legal to assign num to the given row,col
#  Returns a boolean which indicates whether it will be legal to assign
#  num to the given row,col location.
def check_location_is_safe(arr, row, col, num):
    # Check if 'num' is not already placed in current row,
    # current column and current 3x3 box
    return not used_in_row(arr, row, num) and not used_in_col(arr, col, num) and not used_in_box(arr, row - row % 3,
                                                                                                 col - col % 3, num)

# A counter variable to check the number of recursions
# If there are too many recursions, we break the program
counter = 0


# Takes a partially filled-in grid and attempts to assign values to
# all unassigned locations in such a way to meet the requirements
# for Sudoku solution (non-duplication across rows, columns, and boxes)
def solve_sudoku(arr):
    # 'l' is a list variable that keeps the record of row and col in find_empty_location Function
    l = [0, 0]

    global counter
    counter += 1

    if counter > 200000:
        # print("ret falseeee", counter)
        return False

    # If there is no unassigned location, we are done
    if not find_empty_location(arr, l):
        # print("returning true")
        return True

    # Assigning list values to row and col that we got from the above Function
    row = l[0]
    col = l[1]

    # consider digits 1 to 9
    for num in range(1, 10):

        # if looks promising
        if check_location_is_safe(arr, row, col, num):

            # make tentative assignment
            arr[row][col] = num

            # return, if success, ya!
            if solve_sudoku(arr):
                # print("ret true")
                return True

            # failure, unmake & try again
            arr[row][col] = 0

    # this triggers backtracking
    # print("ret false", counter)
    return False


# ----------IMAGE PROCESSING----------


# This method takes in the image, and tries to rectify it
# The four corners of the square might be in any random order
# So what I have done here is ordered the four corners in a particular order
# I have taken this order:
# Top Left and Right
# Then Bottom Left and Right
def rectify(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


# Main driver program for capturing image and processing it
def run():

    # First we initialise the Videocamera
    # IMP: This program requires your computer to have a webcam
    # If you have multiple webcameras, feel free to change the source
    # 0 is for default camera...
    # 1 is for secondary camera...
    cam = cv2.VideoCapture(0)

    # I halted the program for a few seconds to ensure that the Video camera full loads
    # Else, there might be jittery or (laggy) video footage
    time.sleep(3)

    # An inifinite loop to keep displaying the live footage until photo is taken
    while True:
        # Reading the image from the camera
        ret, img = cam.read()
        # Taking the shape...
        # This will be useful later for finding the Sudoku Grid
        height, width, _ = img.shape
        # We convert the image into a grayscale one so that it will be easy to process
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # I now apply an Adaptive Threshold to the image to refine it
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
        # Now we find the contours in the image
        hierarchy, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Now we assume that the largest square in the image is the whole Sudoku Grid
        # This is important
        # If the SUDOKU is not the biggest grid, this program will not work
        # So initially, we assume that there is no biggest grid
        biggest = None
        # Likewise, we assume that the area of the largest grid is 0 (Because it doesn't exist yet)
        max_area = 0
        # Now, we go in the loop
        for i in contours:
            # We try to find the area bounded by the contours
            area = cv2.contourArea(i)
            # I am assuming arbitrarily the number 100
            # It works best
            if area > 100:
                # Find the length
                peri = cv2.arcLength(i, True)
                # Then find the number of sides
                # If the number of sides is 4, then it is a square (rectangle)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                # That is what I check here
                # If the area of the detected square is bigger than the current square, that should be the biggest square
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        # To make that square visible, I draw small circles along the four corners of the square
        cv2.drawContours(img, biggest, -1, (0, 255, 0), 10)
        # This is the font I am using
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Now, I put this text to make it clear how to take a picture
        cv2.putText(img, 'Press spacebar to take picture', (int(height / 2 - 100), int(width / 2 - 200)), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        # This is just the heading of the window
        cv2.imshow('Take Picture', img)
        # I am waiting for keypress
        k = cv2.waitKey(10) & 0xFF
        # If you press spacebar, it will click
        # It will also click if you press enter
        if k == 32 or k == 13:
            break

    # Now we are out of the while loop
    # I now release the camera
    cam.release()
    # Self explanatory....
    # Destroy all the cv2 windows
    cv2.destroyAllWindows()

    # Now, I take the biggest square and rearrange the four corners
    # See the comments in the recity() to understand more
    biggest = rectify(biggest)
    # Now I apply perspective transformations and warping to make the image more clearer
    h = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)
    retval = cv2.getPerspectiveTransform(biggest, h)
    warp = cv2.warpPerspective(gray, retval, (450, 450))

    # I have commented out the below code
    # In case you want to see how the image processing works, just uncomment it
    # It will show you what is currently being done
    # You will just have to press the 'Esc' key to make it disappear

    # while True:
    #     cv2.imshow("Warped", warp)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break

    # Again I have commented out some of the code below
    # I have made the adjustments to fit my webcam quality

    # Don't change this
    updated_img = warp

    # Depending on the quality of the images you are getting, you can tinker around with the values
    # Either uncomment out 1 or 2 and check
    # Whichever works best or you, you can use it
    # 2 works best for me so I have used it
    # Don't use both

    # 1:
    # kernel = np.ones((6, 6), np.uint8)
    # updated_img = cv2.erode(updated_img, kernel, iterations=1)
    # updated_img = cv2.adaptiveThreshold(updated_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 2)

    # 2:
    updated_img = cv2.adaptiveThreshold(updated_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 10)

    # If you want to see which works best in the above case, you can uncomment the below code
    # It will show you the processed image

    # while True:
    #     cv2.imshow("Updated", updated_img)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break

    # Now I just display the captured image
    cv2.putText(warp, 'Press Any Key', (70, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Detected Image', warp)
    cv2.waitKey(0)
    cv2.destroyWindow("Detected Image")

    # I use Tkinter to display a pop up dialog box saying that it may take time processing
    root = tkinter.Tk().withdraw()
    tkinter.messagebox.showinfo("Please wait...", "It might take some time...\nThe OCR is working...")

    # This is the main array for the Sudoku Grid
    Game_board = [[0 for x in range(9)] for y in range(9)]

    # This is the loop in which I try to extract each one of the grids
    for x in range(0, 9):
        for y in range(0, 9):

            # I extract one grid at a time
            morph = updated_img[((50 * x) + 3):((50 * x) + 47), ((50 * y) + 3):((50 * y) + 47)]

            # This is for visualisation purposes only. Comment it if you don't want to see it
            # You have to make a folder called temp if it doesn't exist
            # cv2.imwrite('temp/grid'+str(x)+str(y)+'.jpg', morph)

            # I am using pytesseract for OCR
            # This is the fastest recognition I could get
            # --------------------------------------------
            # If you want more accurate results, you can train a neural net and import that class into here
            # I have tried with the tensorflow MNIST dataset and it works perfectly
            # But the problem is it will take a long time for prediction
            # In case you are okay with that, use the commented loop below this
            # You will have to comment this loop then
            text = pytesseract.image_to_string(morph, lang='eng', config='--psm 6 tessedit_char_whitelist=0123456789')

            # Just checking
            # You can remove the print statement in case you don't want it
            if "1" in text:
                Game_board[x][y] = 1
                print(1, end=" ")
            elif "2" in text:
                Game_board[x][y] = 2
                print(2, end=" ")
            elif "3" in text:
                Game_board[x][y] = 3
                print(3, end=" ")
            elif "4" in text:
                Game_board[x][y] = 4
                print(4, end=" ")
            elif "5" in text:
                Game_board[x][y] = 5
                print(5, end=" ")
            elif "6" in text:
                Game_board[x][y] = 6
                print(6, end=" ")
            elif "7" in text:
                Game_board[x][y] = 7
                print(7, end=" ")
            elif "8" in text:
                Game_board[x][y] = 8
                print(8, end=" ")
            elif "9" in text:
                Game_board[x][y] = 9
                print(9, end=" ")
            else:
                Game_board[x][y] = 0
                print(0, end=" ")
        print()

    # I am displaying the whole 2D Array
    # print("\n", Game_board)

    # Uncomment this code if you want to use the tensorflow model
    # You will need to have the model2.ckpt file

    # for x in range(0, 9):
    #     for y in range(0, 9):
    #         morph = updated_img[((50 * x) + 3):((50 * x) + 47), ((50 * y) + 3):((50 * y) + 47)]
    #         address = 'temp/grid'+str(x)+str(y)+'.jpg'
    #         cv2.imwrite(address, morph)
    #         text = str(predict_2.main(address))
    #         try:
    #             if "1" in text:
    #                 Game_board[x][y] = 1
    #                 print(1, end=" ")
    #             elif "2" in text:
    #                 Game_board[x][y] = 2
    #                 print(2, end=" ")
    #             elif "3" in text:
    #                 Game_board[x][y] = 3
    #                 print(3, end=" ")
    #             elif "4" in text:
    #                 Game_board[x][y] = 4
    #                 print(4, end=" ")
    #             elif "5" in text:
    #                 Game_board[x][y] = 5
    #                 print(5, end=" ")
    #             elif "6" in text:
    #                 Game_board[x][y] = 6
    #                 print(6, end=" ")
    #             elif "7" in text:
    #                 Game_board[x][y] = 7
    #                 print(7, end=" ")
    #             elif "8" in text:
    #                 Game_board[x][y] = 8
    #                 print(8, end=" ")
    #             elif "9" in text:
    #                 Game_board[x][y] = 9
    #                 print(9, end=" ")
    #         except:
    #             Game_board[x][y] = 0
    #             print(0, end=" ")
    #
    # print("\n", Game_board)

    # Display Message that OCR is done
    root = tkinter.Tk().withdraw()
    tkinter.messagebox.showinfo("Solving...", "The OCR is done.\nTrying to solve the Sudoku now...")


    # Now, I am checking whether the Sudoku is solvable
    # If it is solvable, I will print it is solved and work on rendering the pygame Sudoku Grif
    # Else, I will work on rendering the sorry page
    if solve_sudoku(Game_board):
        print("Solved... Displaying")
        # This main is a reference to the function which will render the result
        main(Game_board)
    else:
        print("Cant solve")
        # This is a reference to the sorry function, which will be displayed in case it is not solvable
        sorry()


# ----------GUI PART----------

# Function to display sorry message when sudoku is unsolvable
def sorry():

    # Initialise pygame window
    pygame.init()

    # Window dimensions
    windowWidth = 500
    windowHeight = 500

    # Initialise the display
    gameDisplay = pygame.display.set_mode((windowWidth, windowHeight))

    # Title of the page
    pygame.display.set_caption("Sorry!")

    # Make the screen background white
    gameDisplay.fill(white)

    # Again, the fonts..
    sorryText = pygame.font.SysFont("comicsansms", 30)

    # Main loop to display the message until user quit
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # quit()

        # Make a pygame surface and a pygame rectangle to display the message
        # Once the surface and rectangle are done, align it to the center
        # Then display it (blit)
        TextSurf, TextRect = text_objects("Sorry!", sorryText, black)
        TextRect.center = (windowWidth / 2), (windowHeight / 6)
        gameDisplay.blit(TextSurf, TextRect)

        # Make a pygame surface and a pygame rectangle to display the message
        # Once the surface and rectangle are done, align it to the center
        # Then display it (blit)
        TextSurf, TextRect = text_objects("Either the Sudoku is unsolvable", sorryText, black)
        TextRect.center = (windowWidth / 2), (windowHeight / 4)
        gameDisplay.blit(TextSurf, TextRect)

        # Make a pygame surface and a pygame rectangle to display the message
        # Once the surface and rectangle are done, align it to the center
        # Then display it (blit)
        TextSurf, TextRect = text_objects("OR", sorryText, black)
        TextRect.center = (windowWidth / 2), (windowHeight / 3)
        gameDisplay.blit(TextSurf, TextRect)

        # Make a pygame surface and a pygame rectangle to display the message
        # Once the surface and rectangle are done, align it to the center
        # Then display it (blit)
        TextSurf, TextRect = text_objects("The OCR did not work!", sorryText, black)
        TextRect.center = (windowWidth / 2), (windowHeight / 2)
        gameDisplay.blit(TextSurf, TextRect)
        pygame.display.update()


# Main function which will display the solved sudoku
def main(sudoku):

    # Initialise pygame again
    pygame.init()
    pygame.font.init()

    # Initialise the window
    gameDisplay = pygame.display.set_mode((windowWidth, windowHeight))

    # Set the title
    pygame.display.set_caption("Solved Sudoku")

    # Font....
    numberFont = pygame.font.SysFont("comicsansms", 50)

    # Background..
    gameDisplay.fill(white)
    # drawGrid()

    # Loops
    for x in range(0, windowWidth, cellSize):
        pygame.draw.line(gameDisplay, dull_white, (x, 0), (x, windowHeight))

    for y in range(0, windowHeight, cellSize):
        pygame.draw.line(gameDisplay, dull_white, (0, y), (windowWidth, y))

    for x in range(0, windowWidth, squareSize):
        pygame.draw.line(gameDisplay, black, (x, 0), (x, windowHeight))

    for y in range(0, windowHeight, squareSize):
        pygame.draw.line(gameDisplay, black, (0, y), (windowWidth, y))

    # I put it in a try catch to handle any errors
	try:
		# Main loop to display
		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					# quit()
			# display(sudoku)
			xnow = 5
			ynow = -15
			for x in range(0, 9):
				for y in range(0, 9):
					txt = numberFont.render(str(sudoku[x][y]), True, red)

					gameDisplay.blit(txt, (xnow, ynow))
					xnow += 45
					# print(xnow, ynow)
				ynow += 45
				xnow = 5
				pygame.display.update()
			pygame.display.update()
	except:
		pass


# Function to display the initial screen
def init_scr():

    # I take a temporary variable called intro and let it be True
    intro = True

    # This will execute while intro is true
    while intro:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # quit()

        # Make background white
        gameDisplay.fill(white)

        # Make the font surface objects
        introText = pygame.font.SysFont("comicsansms", 100)
        backgText = pygame.font.SysFont("comicsansms", 40)

        # Add info to the screen
        ss = bytes.fromhex('5375646f6b7520536f6c766572').decode('ascii')
        # Add the text surface and rectangles
        TextSurf, TextRect = text_objects(ss, introText, black)
        # Align it to center
        TextRect.center = (display_width/2, display_height/4)
        # Display it
        gameDisplay.blit(TextSurf, TextRect)

        # Adding more info
        sss = bytes.fromhex('446576656c6f70656420627920526f6368616e2e2e2e').decode('ascii')
        # Add the text surface and rectangles
        TextSurf, TextRect = text_objects(sss, backgText, black)
        # Align it to center
        TextRect.center = (display_width/2, display_height/2)
        # Display it
        gameDisplay.blit(TextSurf, TextRect)

        # Now I try to make the buttons
        # First I get the details of the mouse and its position
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        # I try to make the buttons more interactive
        # I check whether the mouse coordinates are within the coordinates of the button
        # If it is within, I lighten the color to make it look like it's interactive
        # First, check the x coordinates
        if 150 + 100 > mouse[0] > 150 and 450 + 50 > mouse[1] > 450:
            pygame.draw.rect(gameDisplay, bright_green, (150, 450, 100, 50))
            # If the mouse is click
            if click[0] == 1:
                pygame.quit()
                # Display message
                root = tkinter.Tk().withdraw()
                tkinter.messagebox.showinfo("Opening...", "Please wait. Opening your camera...")
                # Run the main sudoku part
                run()
        else:
            pygame.draw.rect(gameDisplay, green, (150, 450, 100, 50))

        # Now check for the y coordinate
        if 550 + 100 > mouse[0] > 550 and 450 + 50 > mouse[1] > 450:
            pygame.draw.rect(gameDisplay, bright_red, (550, 450, 100, 50))
            # Check if it is clicked
            if click[0] == 1:
                pygame.quit()
                # Quit out of the program
                # quit()
        else:
            pygame.draw.rect(gameDisplay, red, (550, 450, 100, 50))

        # There is no method in pygame to make buttons
        # So first I make 2 rectangles
        # The first one is the start
        buttonText = pygame.font.SysFont("comicsansms", 20)
        textSurf, textRect = text_objects("START!", buttonText, dull_white)
        textRect.center = ((150 + (100/2)),(450 + (50/2)))
        gameDisplay.blit(textSurf, textRect)

        # The second one is the quit
        textSurf, textRect = text_objects("QUIT", buttonText, dull_white)
        textRect.center = ((550 + (100/2)), (450 + (50/2)))
        gameDisplay.blit(textSurf, textRect)

        # Finally, update the display
        pygame.display.update()


# This is the function I call from other functions to display text
# It just renders it and returns the text surface and text rectangle
def text_objects(text, font, color):
    textSurface = font.render(text, True, color)
    return textSurface, textSurface.get_rect()

# This is the dimensions of the main screen
display_width = 800
display_height = 600

# Initialise the display
gameDisplay = pygame.display.set_mode((display_width, display_height))
# Set the title
pygame.display.set_caption("Sudoku Solver")

# The main loop to display
while True:
    # initial screen
    init_scr()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            # quit()

    # Update the pygame display
    pygame.display.update()

# Finally, exit pygame and quit python
#pygame.quit()
#quit()
