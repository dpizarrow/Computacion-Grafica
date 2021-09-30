# coding=utf-8
"""Tarea 2: Diego Pizarro W
    Se eligio el modelo 5 de la carpeta carro"""

from os import pipe
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
from  assets_path import getAssetPath

__author__ = "Ivan Sipiran"
__license__ = "MIT"

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.viewPos = np.array([10,10,10])
        self.camUp = np.array([0, 1, 0])
        self.distance = 10


controller = Controller()

def setPlot(pipeline, mvpPipeline):
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)

    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 5, 5, 5)
    
    glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.01)

def setView(pipeline, mvpPipeline):
    view = tr.lookAt(
            controller.viewPos,
            np.array([0,0,0]),
            controller.camUp
        )

    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "viewPosition"), controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])
    

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
    
    elif key == glfw.KEY_1:
        controller.viewPos = np.array([controller.distance,controller.distance,controller.distance]) #Vista diagonal 1
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_2:
        controller.viewPos = np.array([0,0,controller.distance]) #Vista frontal
        controller.camUp = np.array([0,1,0])

    elif key == glfw.KEY_3:
        controller.viewPos = np.array([controller.distance,0,controller.distance]) #Vista lateral
        controller.camUp = np.array([0,1,0])

    elif key == glfw.KEY_4:
        controller.viewPos = np.array([0,controller.distance,0]) #Vista superior
        controller.camUp = np.array([1,0,0])
    
    elif key == glfw.KEY_5:
        controller.viewPos = np.array([controller.distance,controller.distance,-controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_6:
        controller.viewPos = np.array([-controller.distance,controller.distance,-controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_7:
        controller.viewPos = np.array([-controller.distance,controller.distance,controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    else:
        print('Unknown key')

def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

#NOTA: Aqui creas tu escena. En escencia, sólo tendrías que modificar esta función.
def createScene(pipeline):
    
    # Creamos un cilindro que representara los neumaticos de la camioneta
    wheel = createGPUShape(pipeline, bs.createColorCylinderTarea2(0.0,0.0,0.0))
    
    # Creamos un cilindro que representara las llantas
    wheelcover = createGPUShape(pipeline, bs.createColorCylinderTarea2(1.0, 1.0, 1.0))
    
    # Creamos un cubo que se usara para las distintas partes del chassis de la camioneta
    body = createGPUShape(pipeline, bs.createColorCubeTarea2(0.0, 0.0, 1.0))

    # Creamos un cubo que se usara para los detalles de la parte delantera de la camioneta
    details = createGPUShape(pipeline, bs.createColorCubeTarea2(0.411, 0.411, 0.411))

    # Rueda delantera izquierda
    frontLeft = sg.SceneGraphNode('frontLeft')
    frontLeft.transform = tr.matmul([tr.translate(0.8, 0.5, -0.5) , tr.scale(0.25, 0.25, 0.1), tr.rotationX(np.pi/2)])
    frontLeft.childs += [wheel]
    
    # Rueda delantera derecha
    frontRight = sg.SceneGraphNode('frontRight')
    frontRight.transform = tr.matmul([tr.translate(0.8, 0.5, 0.5) , tr.scale(0.25, 0.25, 0.1), tr.rotationX(np.pi/2)])
    frontRight.childs += [wheel]

    # Rueda trasera izquierda
    rearLeft = sg.SceneGraphNode('rearLeft')
    rearLeft.transform  = tr.matmul([tr.translate(-0.7, 0.5, -0.5) , tr.scale(0.25, 0.25, 0.1), tr.rotationX(np.pi/2)])
    rearLeft.childs += [wheel]

    # Rueda trasera derecha
    rearRight = sg.SceneGraphNode('rearRight')
    rearRight.transform  = tr.matmul([tr.translate(-0.7, 0.5, 0.5) , tr.scale(0.25, 0.25, 0.1), tr.rotationX(np.pi/2)])
    rearRight.childs += [wheel]

    # Parte central de la camioneta
    middleBody = sg.SceneGraphNode('middleBody')
    middleBody.transform = tr.matmul([tr.translate(0.35, 0.8, 0.0), tr.scale(0.65, 0.1, 0.5), tr.rotationY(np.pi/2)])
    middleBody.childs += [body]

    # Parte derecha del pickup de la camioneta
    rightPickup = sg.SceneGraphNode('rightPickup')
    rightPickup.transform = tr.matmul([tr.translate(-0.2, 0.7, 0.49), tr.scale(1.2, 0.2, 0.04)])
    rightPickup.childs += [body]

    # Parte izquierda del pickup de la camioneta
    leftPickup = sg.SceneGraphNode('leftPickup')
    leftPickup.transform = tr.matmul([tr.translate(-0.2, 0.7, -0.49), tr.scale(1.2, 0.2, 0.04)])
    leftPickup.childs += [body]

    # Parte trasera del pickup
    backPickup = sg.SceneGraphNode('backPickup')
    backPickup.transform = tr.matmul([tr.rotationY(np.pi/2), tr.translate(-0.02, 0.7, -1.4), tr.scale(0.5, 0.2, 0.025)])
    backPickup.childs += [body]

    # Parachoque de la camioneta
    grill = sg.SceneGraphNode('grill')
    grill.transform = tr.matmul([tr.rotationY(np.pi/2), tr.translate(0.0, 0.7, 1), tr.scale(0.55, 0.2, 0.025)])
    grill.childs += [body]

    # Piso del pickup
    bottomPickup = sg.SceneGraphNode('bottomPickup')
    bottomPickup.transform = tr.matmul([tr.rotationX(np.pi), tr.translate(-0.65, -0.6, 0), tr.scale(0.7, 0.05, 0.45)])
    bottomPickup.childs += [body]

    # Parabrisas de la camioneta
    windshield = sg.SceneGraphNode('windshield')
    windshield.transform = tr.matmul([tr.rotationY(np.pi/2), tr.rotationX(-np.pi/4), tr.translate(0.0, 0.4, 1.0), tr.scale(0.5, 0.2, 0.025)])
    windshield.childs += [body]

    # Parte trasera de la camioneta, se conectara con el techo
    backBody = sg.SceneGraphNode('backBody')
    backBody.transform = tr.matmul([tr.rotationY(np.pi/2), tr.translate(-0.02, 1.0, -0.3), tr.scale(0.5, 0.14, 0.025)])
    backBody.childs += [body]

    # Techo de la camioneta
    roof = sg.SceneGraphNode('roof')
    roof.transform = tr.matmul([tr.rotationX(np.pi), tr.translate(-0.01, -1.15, 0.0), tr.scale(0.33, 0.03, 0.45)])
    roof.childs += [body]
    
    # Llanta delantera izquierda
    flwheelCover = sg.SceneGraphNode('flwheelCover')
    flwheelCover.transform = tr.matmul([tr.translate(0.8, 0.5, -0.6) , tr.scale(0.15, 0.15, 0.01), tr.rotationX(np.pi/2)])
    flwheelCover.childs += [wheelcover]

    # Llanta delantera derecha
    frwheelCover = sg.SceneGraphNode('frwheelCover')
    frwheelCover.transform = tr.matmul([tr.translate(0.8, 0.5, 0.6) , tr.scale(0.15, 0.15, 0.01), tr.rotationX(np.pi/2)])
    frwheelCover.childs += [wheelcover]

    # Llanta trasera izquierda
    blwheelCover = sg.SceneGraphNode('blwheelCover')
    blwheelCover.transform = tr.matmul([tr.translate(-0.7, 0.5, -0.6) , tr.scale(0.15, 0.15, 0.01), tr.rotationX(np.pi/2)])
    blwheelCover.childs += [wheelcover]

    # Llanta trasera derecha
    brwheelCover = sg.SceneGraphNode('brwheelCover')
    brwheelCover.transform = tr.matmul([tr.translate(-0.7, 0.5, 0.6) , tr.scale(0.15, 0.15, 0.01), tr.rotationX(np.pi/2)])
    brwheelCover.childs += [wheelcover]

    # Base para los detalles del parachoque
    bumper = sg.SceneGraphNode('bumper')
    bumper.transform = tr.matmul([tr.rotationY(np.pi/2), tr.translate(-0.02, 0.7, 1.05), tr.scale(0.5, 0.1, 0.001)])
    bumper.childs += [details]

    # Parachoque trasero
    rearBumper = sg.SceneGraphNode('rearBumper')
    rearBumper.transform = tr.matmul([tr.rotationY(np.pi/2), tr.translate(-0.02, 0.5, -1.43), tr.scale(0.5, 0.1, 0.001)])
    rearBumper.childs += [details]

    # Creamos el grafo de escena y los nodos 
    scene = sg.SceneGraphNode('system')
    
    # Nodo para los neumaticos 
    wheelsNode = sg.SceneGraphNode('wheels')
    
    # Nodo para el chassis
    bodyNode = sg.SceneGraphNode('body')
    
    # Nodo para las llantas
    coverNode = sg.SceneGraphNode('covers')

    # Nodo para los detalles del parachoque
    bumperNode = sg.SceneGraphNode('bumper')
    
    wheelsNode.childs += [frontLeft, frontRight, rearLeft, rearRight]
    bodyNode.childs += [middleBody, rightPickup, leftPickup, backPickup, bottomPickup, grill, windshield, backBody, roof]
    coverNode.childs += [flwheelCover, frwheelCover, blwheelCover, brwheelCover]
    bumperNode.childs += [bumper, rearBumper]
    scene.childs += [wheelsNode, bodyNode, coverNode, bumperNode]
    
    scene.transform = tr.matmul([tr.uniformScale(2)])
    return scene

if __name__ == "__main__":

    window = None
    
    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 2"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    pipeline = ls.SimpleGouraudShaderProgram()
    
    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    mvpPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

    #NOTA: Aqui creas un objeto con tu escena
    dibujo = createScene(pipeline)

    setPlot(pipeline, mvpPipeline)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        setView(pipeline, mvpPipeline)

        if controller.showAxis:
            glUseProgram(mvpPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            mvpPipeline.drawCall(gpuAxis, GL_LINES)

        #NOTA: Aquí dibujas tu objeto de escena
        glUseProgram(pipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, pipeline, "model")
        

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()
    

    glfw.terminate()