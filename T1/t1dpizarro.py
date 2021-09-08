import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy

__author__ = "Ivan Sipiran"
__license__ = "MIT"

# We will use 32 bits data, so an integer has 4 bytes
# 1 byte = 8 bits
SIZE_IN_BYTES = 4

# Esta funcion crea un fondo negro, sobre la cual vamos a poner las casillas blancas
def crear_fondo():
    fondo = [
        -0.75, -0.75, 0, 0, 0, 0,
        0.75, -0.75, 0, 0, 0, 0, 
        0.75, 0.75, 0, 0, 0, 0,
        0.75, 0.75, 0, 0, 0, 0,
        -0.75, 0.75, 0, 0, 0, 0,
        -0.75, -0.75, 0, 0, 0, 0
    ]
    return numpy.array(fondo, dtype=numpy.float32)


# crea un cuadrado de dimensiones 0.1875 x 0.1875
# x e y son las coordenadas del vertice inferior izquierdo del cuadrado
def crear_cuadrado(x, y, r, g, b):
    cuadrado = [
        x, y, 0.0, r, g, b,
        x + 0.1875, y, 0.0, r, g, b,
        x + 0.1875, y + 0.1875, 0.0,  r, g, b, 
        x + 0.1875, y + 0.1875, 0.0, r, g, b, 
        x, y + 0.1875, 0.0, r, g, b, 
        x, y, 0.0, r, g, b
    ]
    return cuadrado


# crea el arreglo de las casillas blancas
def crear_tablero():
    output = []
    for i in range(8):
        # en las filas pares el primer cuadrado blanco empieza en (-0.75, -0.75)
        if i % 2 == 0:
            # creamos 4 casillas blancas a una distancia de 0.375 cada una
            for j in range(4):
                output.extend(crear_cuadrado(-0.75 + 0.375 * j, -0.75 + 0.1875 * i, 1, 1, 1))
        # en las filas impares el primer cuadrado blanco empieza en (-0.5625, -0.5625)
        else:
            # creamos 4 casillas blancas a una distancia de 0.375 cada una
            for j in range(4):
                output.extend(crear_cuadrado(-0.5625 + 0.375 * j, -0.75 + 0.1875 * i, 1, 1, 1))
    
    return numpy.array(output, dtype=numpy.float32)

def crear_dama(x,y,r,g,b,radius):
    
    circle = []
    for angle in range(0,360,10):
        circle.extend([x, y, 0.0, r, g, b])
        circle.extend([x+numpy.cos(numpy.radians(angle))*radius, 
                       y+numpy.sin(numpy.radians(angle))*radius, 
                       0.0, r, g, b])
        circle.extend([x+numpy.cos(numpy.radians(angle+10))*radius, 
                       y+numpy.sin(numpy.radians(angle+10))*radius, 
                       0.0, r, g, b])
    
    return numpy.array(circle, dtype = numpy.float32)

# Crea las 24 damas en sus posiciones correctas
def concat_damas():
    damas_red = numpy.array([], dtype=numpy.float32)
    damas_blue = numpy.array([], dtype=numpy.float32)

    # creamos 3 filas de damas azules
    for i in range(0, 4):
        damas_blue = numpy.append(damas_blue, crear_dama(-0.46875 + 0.375 * i, -0.65625, 0.0, 0.0, 1.0, 0.08))
        damas_blue = numpy.append(damas_blue, crear_dama(-0.65625 + 0.375 * i, -0.65625 + 0.1875, 0.0, 0.0, 1.0, 0.08))
        damas_blue = numpy.append(damas_blue, crear_dama(-0.46875 + 0.375 * i, -0.65625 + 0.1875 * 2, 0.0, 0.0, 1.0, 0.08))
    
    # creamos 3 filas de damas rojas
    for i in range(0, 4):
        damas_red = numpy.append(damas_red, crear_dama(-0.65625 + 0.375 * i,-0.65625 + 0.1875 * 5, 1.0, 0.0, 0.0, 0.08))
        damas_red = numpy.append(damas_red, crear_dama(-0.46875 + 0.375 * i,-0.65625 + 0.1875 * 6, 1.0, 0.0, 0.0, 0.08))
        damas_red = numpy.append(damas_red, crear_dama(-0.65625 + 0.375 * i,-0.65625 + 0.1875 * 7, 1.0, 0.0, 0.0, 0.08))
    
    return numpy.append(damas_red, damas_blue)


if __name__ == "__main__":

    window = None
    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Tarea 1 - Diego Pizarro W.", None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    fondo = crear_fondo()
    tablero = crear_tablero()
    dama = concat_damas()
    
    # Defining shaders for our pipeline
    vertex_shader = """
    #version 330
    in vec3 position;
    in vec3 color;

    out vec3 newColor;
    void main()
    {
        gl_Position = vec4(position, 1.0f);
        newColor = color;
    }
    """

    fragment_shader = """
    #version 330
    in vec3 newColor;

    out vec4 outColor;
    void main()
    {
        outColor = vec4(newColor, 1.0f);
    }
    """

    # Binding artificial vertex array object for validation
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # Assembling the shader program (pipeline) with both shaders
    shaderProgram = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    # Each shape must be attached to a Vertex Buffer Object (VBO)
    
    vboFondo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vboFondo)
    glBufferData(GL_ARRAY_BUFFER, len(fondo) * SIZE_IN_BYTES, fondo, GL_STATIC_DRAW)

    vboTablero = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vboTablero)
    glBufferData(GL_ARRAY_BUFFER, len(tablero) * SIZE_IN_BYTES, tablero, GL_STATIC_DRAW)

    vboDama = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vboDama)
    glBufferData(GL_ARRAY_BUFFER, len(dama) * SIZE_IN_BYTES, dama, GL_STATIC_DRAW)

    # Telling OpenGL to use our shader program
    glUseProgram(shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.5,0.5, 0.5, 1.0)

    glClear(GL_COLOR_BUFFER_BIT)

    # Setup del Fondo
    glBindBuffer(GL_ARRAY_BUFFER, vboFondo)
    position = glGetAttribLocation(shaderProgram, "position") # Buscar la variable position en el shader, solo es necesario hacer una vez por ShaderProgram
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shaderProgram, "color") # Buscar la variable color en el shader, solo es necesario hacer una vez por ShaderProgram
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
    
    # Dibujamos el fondo
    glDrawArrays(GL_TRIANGLES, 0, int(len(fondo)/6))

    # Setup de los cuadrados blancos del tablero
    glBindBuffer(GL_ARRAY_BUFFER, vboTablero)
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
    
    # Dibujamos los cuadrados
    glDrawArrays(GL_TRIANGLES, 0, int(len(tablero)/6))   
    
    # Setup de las dams
    glBindBuffer(GL_ARRAY_BUFFER, vboDama)
    position = glGetAttribLocation(shaderProgram, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shaderProgram, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
    
    # Dibujamos las damas
    glDrawArrays(GL_TRIANGLES, 0, int(len(dama)/6))

    # Moving our draw to the active color buffer
    glfw.swap_buffers(window)

    # Waiting to close the window
    while not glfw.window_should_close(window):

        # Getting events from GLFW
        glfw.poll_events()
        
    glfw.terminate()