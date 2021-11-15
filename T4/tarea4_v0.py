# coding=utf-8
"""Tarea 4"""


import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transformations as tr
import basic_shapes as bs
import scene_graph as sg
import easy_shaders as es
import lighting_shaders as ls
import performance_monitor as pm
from assets_path import getAssetPath
from auxiliarT4 import *
from operator import add


#Este código está basado en el código de Valentina Aguilar.

__author__ = "Valentina Aguilar  - Ivan Sipiran - Juan Pablo Alvira"


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.X = 2.0 #posicion X de donde esta el auto
        self.Y = -0.037409 #posicion Y de donde esta el auto
        self.Z = 5.0 #posicion Z de donde esta el auto
        #lo siguiente se creo para poder usar coordenadas esfericas
        self.cameraPhiAngle = -np.pi/4 #inclinacion de la camara 
        self.cameraThetaAngle = np.pi/2 #rotacion con respecto al eje y
        self.r = 2 #radio

#TAREA4: Esta clase contiene todos los parámetros de una luz Spotlight. Sirve principalmente para tener
# un orden sobre los atributos de las luces
class Spotlight:
    def __init__(self):
        self.ambient = np.array([0,0,0])
        self.diffuse = np.array([0,0,0])
        self.specular = np.array([0,0,0])
        self.constant = 0
        self.linear = 0
        self.quadratic = 0
        self.position = np.array([0,0,0])
        self.direction = np.array([0,0,0])
        self.cutOff = 0
        self.outerCutOff = 0

controller = Controller()

#TAREA4: aquí se crea el pool de luces spotlight (como un diccionario)
spotlightsPool = dict()


coord_X = 0 
coord_Z = 0
angulo = 0


def generateT(t):
    return np.array([[1, t, t ** 2, t ** 3]]).T


def hermiteMatrix(P1, P2, T1, T2):
    # Generate a matrix concatenating the columns
    G = np.concatenate((P1, P2, T1, T2), axis=1)

    # Hermite base matrix is a constant
    Mh = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])

    return np.matmul(G, Mh)

def evalCurve(M, N):
    # The parameter t should move between 0 and 1
    ts = np.linspace(0.0, 1.0, N)

    # The computed value in R3 for each sample will be stored here
    curve = np.ndarray(shape=(N, 3), dtype=float)

    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T # x, y, z

    return curve

# En total hay 4 luces, uno para cada foco de los autos
def setLights():
    
    # Luz izquierda del auto que es controlado con las teclas

    spot1 = Spotlight()
    spot1.ambient = np.array([0.0, 0.0, 0.0])
    spot1.diffuse = np.array([1.0, 1.0, 1.0])
    spot1.specular = np.array([1.0, 1.0, 1.0])
    spot1.constant = 1
    spot1.linear = 0.09
    spot1.quadratic = 0.032
    spot1.position = np.array([0,0,0]) 
    spot1.direction = np.array([0, 0, -1])
    spot1.cutOff = np.cos(np.radians(12.5)) 
    spot1.outerCutOff = np.cos(np.radians(30))                     
    spotlightsPool['leftLight'] = spot1 

    # Luz derecha del auto que es controlado con las teclas

    spot2 = Spotlight()
    spot2.ambient = np.array([0.0, 0.0, 0.0])
    spot2.diffuse = np.array([1.0, 1.0, 1.0])
    spot2.specular = np.array([1.0, 1.0, 1.0])
    spot2.constant = 1
    spot2.linear = 0.09
    spot2.quadratic = 0.032
    spot2.position = np.array([-2, 5, 0]) 
    spot2.direction = np.array([0, -1, 0]) 
    spot2.cutOff = np.cos(np.radians(12.5))
    spot2.outerCutOff = np.cos(np.radians(30)) 
    spotlightsPool['rightLight'] = spot2 

    # Luz derecha del auto que se mueve solo

    spot3 = Spotlight()
    spot3.ambient = np.array([0.0, 0.0, 0.0])
    spot3.diffuse = np.array([1.0, 1.0, 1.0])
    spot3.specular = np.array([1.0, 1.0, 1.0])
    spot3.constant = 1
    spot3.linear = 0.09
    spot3.quadratic = 0.032
    spot3.position = np.array([-2, 5, 0]) 
    spot3.direction = np.array([0, -1, 0]) 
    spot3.cutOff = np.cos(np.radians(12.5))
    spot3.outerCutOff = np.cos(np.radians(30)) 
    spotlightsPool['car2right'] = spot3  

    # Luz izquierda del auto que se mueve solo

    spot4 = Spotlight()
    spot4.ambient = np.array([0.0, 0.0, 0.0])
    spot4.diffuse = np.array([1.0, 1.0, 1.0])
    spot4.specular = np.array([1.0, 1.0, 1.0])
    spot4.constant = 1
    spot4.linear = 0.09
    spot4.quadratic = 0.032
    spot4.position = np.array([-2, 5, 0]) 
    spot4.direction = np.array([0, -1, 0]) 
    spot4.cutOff = np.cos(np.radians(12.5))
    spot4.outerCutOff = np.cos(np.radians(30)) 
    spotlightsPool['car2left'] = spot4  


#TAREA4: modificamos esta función para poder configurar todas las luces del pool
def setPlot(texPipeline, axisPipeline, lightPipeline):
    projection = tr.perspective(60, float(width)/float(height), 0.1, 100) #el primer parametro se cambia a 60 para que se vea más escena

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    #TAREA4: Como tenemos 2 shaders con múltiples luces, tenemos que enviar toda esa información a cada shader
    #TAREA4: Primero al shader de color
    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    #TAREA4: Enviamos la información de la luz puntual y del material
    #TAREA4: La luz puntual está desactivada por defecto (ya que su componente ambiente es 0.0, 0.0, 0.0), pero pueden usarla
    # para añadir más realismo a la escena
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].ambient"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].diffuse"), 0.0, 0.0, 0.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].specular"), 0.0, 0.0, 0.0)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].constant"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].linear"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].quadratic"), 0.01)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].position"), 5, 5, 5)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "material.ambient"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "material.diffuse"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "material.specular"), 1.0, 1.0, 1.0)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "material.shininess"), 32)

    #TAREA4: Aprovechamos que las luces spotlight están almacenadas en el diccionario para mandarlas al shader
    for i, (k,v) in enumerate(spotlightsPool.items()):
        baseString = "spotLights[" + str(i) + "]."
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "ambient"), 1, v.ambient)
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "diffuse"), 1, v.diffuse)
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "specular"), 1, v.specular)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "constant"), v.constant)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "linear"), 0.09)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "quadratic"), 0.032)
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "position"), 1, v.position)
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "direction"), 1, v.direction)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "cutOff"), v.cutOff)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "outerCutOff"), v.outerCutOff)

    #TAREA4: Ahora repetimos todo el proceso para el shader de texturas con mútiples luces
    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    

    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].ambient"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].diffuse"), 0.0, 0.0, 0.0)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].specular"), 0.0, 0.0, 0.0)
    glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].constant"), 0.1)
    glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].linear"), 0.1)
    glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].quadratic"), 0.01)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].position"), 5, 5, 5)

    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "material.ambient"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "material.diffuse"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "material.specular"), 1.0, 1.0, 1.0)
    glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, "material.shininess"), 32)

    for i, (k,v) in enumerate(spotlightsPool.items()):
        baseString = "spotLights[" + str(i) + "]."
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "ambient"), 1, v.ambient)
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "diffuse"), 1, v.diffuse)
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "specular"), 1, v.specular)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "constant"), v.constant)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "linear"), 0.09)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "quadratic"), 0.032)
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "position"), 1, v.position)
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "direction"), 1, v.direction)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "cutOff"), v.cutOff)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "outerCutOff"), v.outerCutOff)

#TAREA4: Esta función controla la cámara
def setView(texPipeline, axisPipeline, lightPipeline):
    #la idea de usar coordenadas esfericas para la camara fue extraida del auxiliar 6
    #como el auto reposa en el plano XZ, no sera necesaria la coordenada Y esferica.
    Xesf = controller.r * np.sin(controller.cameraPhiAngle)*np.cos(controller.cameraThetaAngle) #coordenada X esferica
    Zesf = controller.r * np.sin(controller.cameraPhiAngle)*np.sin(controller.cameraThetaAngle) #coordenada Y esferica

    viewPos = np.array([controller.X-Xesf,0.5,controller.Z-Zesf])
    view = tr.lookAt(
            viewPos, #eye
            np.array([controller.X,controller.Y,controller.Z]),     #at
            np.array([0, 1, 0])   #up
        )

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    
    

def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    
if __name__ == "__main__":

    window = None

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 4 Diego Pizarro W."
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    #TAREA4: Se usan los shaders de múltiples luces
    axisPipeline = es.SimpleModelViewProjectionShaderProgram()
    texPipeline = ls.MultipleLightTexturePhongShaderProgram()
    lightPipeline = ls.MultipleLightPhongShaderProgram()
    
    # Telling OpenGL to use our shader program
    glUseProgram(axisPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    axisPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

    #NOTA: Aqui creas un objeto con tu escena
    #TAREA4: Se cargan las texturas y se configuran las luces
    loadTextures()
    setLights()

    dibujo = createStaticScene(texPipeline)
    car = createCarScene(lightPipeline)
    
    # Creamos el segundo auto que se movera solo por la pista
    
    car2 = createCarScene(lightPipeline) 

    
    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    #parametro iniciales
    t0 = glfw.get_time()
    
    # Creamos la curva por la cual se movera el segundo auto

    # Creamos la primera parte recta de la curva

    N = 200
    P1 = np.array([[1.5,0.037409,5.5]]).T
    P2 = np.array([[1.5,0.037409,-4.5]]).T
    T1 = np.array([[0, 0.037409, 0]]).T
    T2 = np.array([[0, 0.037409, 0]]).T

    mat1 = hermiteMatrix(P1, P2, T1, T2)
    hermiteCurve1 = evalCurve(mat1, N)

    # Creamos la segunda curva recta
    
    P3 = np.array([[-2.5,0.037409,-4.5]]).T
    P4 = np.array([[-2.5,0.037409,5.5]]).T
    T3 = np.array([[0, 0.037409, 0]]).T
    T4 = np.array([[0, 0.037409, 0]]).T

    mat2 = hermiteMatrix(P3, P4, T3, T4)
    hermiteCurve2 = evalCurve(mat2, N)

    # Primera curva
    
    P5 = np.array([[1.5,0.037409,-4.5]]).T
    P6 = np.array([[-2.5,0.037409,-4.5]]).T
    T5 = np.array([[0, 0, -8]]).T
    T6 = np.array([[0, 0, 8]]).T

    mat3 = hermiteMatrix(P5, P6, T5, T6)
    hermiteCurve3 = evalCurve(mat3, N)

    # Segunda curva
    
    P7 = np.array([[-2.5,0.037409,5.5]]).T
    P8 = np.array([[1.5,0.037409,5.5]]).T
    T7 = np.array([[0, 0, 8]]).T
    T8 = np.array([[0, 0, -8]]).T

    mat4 = hermiteMatrix(P7,P8,T7,T8)
    hermiteCurve4 = evalCurve(mat4, N)

    # Concatenamos todas las curvas
    
    C = np.concatenate((hermiteCurve1,hermiteCurve3,hermiteCurve2,hermiteCurve4))
    

    # Definimos los parametros que nos ayudaran a controlar el movimiento del auto por la curva
    
    step = 0
    angle2 = 0
    
    # Definimos una variable que nos ayudara a controlar la velocidad del auto

    t = 0

    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        #Se obtiene una diferencia de tiempo con respecto a la iteracion anterior.
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        
        #TAREA4: Se manejan las teclas de la animación
        #ir hacia adelante 
        if(glfw.get_key(window, glfw.KEY_W) == glfw.PRESS):
            controller.X -= 1.5 * dt * np.sin(angulo) #avanza la camara
            controller.Z -= 1.5 * dt * np.cos(angulo) #avanza la camara
            coord_X -= 1.5 * dt * np.sin(angulo) #avanza el auto
            coord_Z -= 1.5 * dt * np.cos(angulo) #avanza el auto

        #ir hacia atras
        if(glfw.get_key(window, glfw.KEY_S) == glfw.PRESS):
            controller.X += 1.5 * dt * np.sin(angulo) #retrocede la camara
            controller.Z += 1.5 * dt * np.cos(angulo) #retrocede la cmara
            coord_X += 1.5 * dt * np.sin(angulo) #retrocede el auto
            coord_Z += 1.5 * dt * np.cos(angulo) #retrocede el auto

        #ir hacia la izquierda
        if(glfw.get_key(window, glfw.KEY_A) and ( glfw.get_key(window, glfw.KEY_W) or glfw.get_key(window, glfw.KEY_S)) == glfw.PRESS):
            controller.cameraThetaAngle -= dt  #camara se gira a la izquierda
            angulo += dt #auto gira a la izquierda

        #ir hacia la derecha
        if(glfw.get_key(window, glfw.KEY_D) and ( glfw.get_key(window, glfw.KEY_W) or glfw.get_key(window, glfw.KEY_S)) == glfw.PRESS):
            controller.cameraThetaAngle += dt #camara se gira a la derecha
            angulo -= dt #auto gira a la derecha

        #Defining car position, where the car is looking at as a vector and a vector that is perpendicular to this last one to properly orientate the lights.
        car_pos = np.array([coord_X+2,-0.037409,coord_Z+5])
        car_at = np.array([car_pos[0] + np.cos(controller.cameraThetaAngle),
                                car_pos[1], 
                                car_pos[2] + np.sin(controller.cameraThetaAngle)])

        car_at_right = np.array([car_pos[0] + np.cos(controller.cameraThetaAngle + np.pi/2),
                                car_pos[1], 
                                car_pos[2] + np.sin(controller.cameraThetaAngle + np.pi/2)])

        # Vectores para las luces del auto controlado con  las teclas
        
        carLight = car_pos - car_at
        carLight = carLight/np.linalg.norm(carLight)

        # Normalizamos el vector

        carLightR = car_pos - car_at_right
        carLightR = carLightR/np.linalg.norm(carLightR)

        # Fijamos las posiciones y direcciones de los vectores de las luces del auto
        
        spotlightsPool["leftLight"].direction = carLight
        spotlightsPool["leftLight"].position = np.array([car_pos - 0.1*carLightR + 0.4*carLight]) + np.array([0,0.1,0])

        spotlightsPool["rightLight"].direction = carLight
        spotlightsPool["rightLight"].position = np.array([car_pos + 0.1*carLightR + 0.4*carLight]) + np.array([0,0.1,0])

    
        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        #TAREA4: Ojo aquí! Se configura la cámara y el dibujo en cada iteración. Esto es porque necesitamos que en cada iteración
        # las luces de los faros de los carros se actualicen en posición y dirección
        setView(texPipeline, axisPipeline, lightPipeline)
        setPlot(texPipeline, axisPipeline,lightPipeline)

        if controller.showAxis:
            glUseProgram(axisPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            axisPipeline.drawCall(gpuAxis, GL_LINES)

        #NOTA: Aquí dibujas tu objeto de escena
        glUseProgram(texPipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, texPipeline, "model")
        
        glUseProgram(lightPipeline.shaderProgram)

        #aqui se mueve el auto
        sg.drawSceneGraphNode(car, lightPipeline, "model")
        Auto = sg.findNode(car,'system-car')
        Auto.transform = tr.matmul([tr.translate(coord_X+2,-0.037409,coord_Z+5),tr.rotationY(np.pi+angulo),tr.rotationY(-np.pi),tr.translate(-2,0.037409,-5)])
        
        
        if step > N*4-1:
            step = 0
        if step < N*4-1:
            angle2 = np.arctan2(C[step+1,0]-C[step,0], C[step+1,2]-C[step,2])
        else:
            angle2 = np.arctan2(C[0,0]-C[step,0],C[0,2]-C[step,2])

        # Dibujamos el auto que se mueve solo
        
        sg.drawSceneGraphNode(car2, lightPipeline, "model")

        # Posicion del segundo auto
        
        car2pos = np.array([C[step][0],C[step][1],C[step][2]]) + np.array([0.5,0,0])
        
        # Buscamos el segundo auto y le aplicamos la transformacion segun la forma de la curva
        
        car2node = sg.findNode(car2,'system-car')
        car2node.transform = tr.matmul([tr.translate(car2pos[0],car2pos[1],car2pos[2]),tr.rotationY(angle2+np.pi),tr.translate(-2,0,-5)])
        
        #Vector que apunta hacia adonde esta mirando el auto que se mueve solo 
        
        car2at = np.array([car2pos[0] - np.cos(angle2-np.pi/2),
                                car2pos[1], 
                                car2pos[2] + np.sin(angle2-np.pi/2)])
        
        # Vector perpendicular al car2_at
     
        car2at_right = np.array([car2pos[0] - np.cos(angle2),
                                car2pos[1], 
                                car2pos[2] + np.sin(angle2)])

        # Normalizamos
        car2Light = car2pos - car2at
        car2Light = car2Light/np.linalg.norm(car2Light)

        car2LightR = car2pos - car2at_right
        car2LightR = car2LightR/np.linalg.norm(car2LightR)
        
        # Al igual que para el otro auto, definimos las posiciones y direcciones de las luces
        
        spotlightsPool["car2left"].direction = car2Light
        spotlightsPool["car2left"].position = np.array([car2pos - 0.1*car2LightR + 0.4*car2Light]) + np.array([0,0.1,0])

        spotlightsPool["car2right"].direction = car2Light
        spotlightsPool["car2right"].position = np.array([car2pos + 0.1*car2LightR + 0.4*car2Light]) + np.array([0,0.1,0])


        # Modoficamos los valores del auto para que no vaya tan rapido

        t = (t+1) % 5
        
        if((t) == 4):
            step +=1    

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()
    

    glfw.terminate()