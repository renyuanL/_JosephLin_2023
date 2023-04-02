# using turtle graphics to draw a smiley face

import turtle

# draw a circle
def drawCircle(t, x, y, r):
    t.penup()
    t.goto(x + r, y)
    t.pendown()
    t.circle(r)

# draw an eye
def drawEye(t, x, y):
    t.penup()
    t.goto(x, y)
    t.pendown()
    t.begin_fill()
    t.circle(10)
    t.end_fill()

# draw a smile
def drawSmile(t, x, y):
    t.penup()
    t.goto(x, y)
    t.pendown()
    t.setheading(-45)
    t.circle(50, 90)
    t.setheading(0)
    t.forward(20)
    t.setheading(90)
    t.forward(20)

def main():
    t= turtle.Turtle()
    t.speed(0)
    t.hideturtle()
    t.pensize(5)
    t.color('red', 'yellow')
    t.begin_fill()
    drawCircle(t, 0, 0, 100)
    t.end_fill()
    drawEye(t, -40, 50)
    drawEye(t, 40, 50)
    drawSmile(t, 0, 20)
    turtle.done()

main()

# Path: ryEx002.py

