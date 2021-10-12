# coding=utf-8
"""Tarea 3"""

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
from assets_path import getAssetPath
from operator import add

__author__ = "Ivan Sipiran"
__license__ = "MIT"

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        # INITIAL CAMERA POS
        #self.viewPos = np.array([2.0,0.3,6.0])
        #self.at = np.array([2.0,-0.037409,3])
        self.viewPos = np.array([5, 5, 5])
        self.at = np.array([0, 0, 0])
        self.camUp = np.array([0, 1, 0])
        self.distance = 20


controller = Controller()

def setPlot(texPipeline, axisPipeline, lightPipeline):
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "lightPosition"), 5, 5, 5)
    
    glUniform1ui(glGetUniformLocation(lightPipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "constantAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

def setView(texPipeline, axisPipeline, lightPipeline):
    view = tr.lookAt(
            controller.viewPos,
            controller.at,
            controller.camUp
        )

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "viewPosition"), controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])
    

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

    elif key == glfw.KEY_9:
        controller.viewPos = np.array([0.01,20,0]) #Vista diagonal 2
        controller.at = np.array([0,0,0])
        controller.camUp = np.array([0,1,0])
    
    else:
        print('Unknown key')

def createOFFShape(pipeline, filename, r,g, b):
    shape = readOFF(getAssetPath(filename), (r, g, b))
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def readOFF(filename, color):
    vertices = []
    normals= []
    faces = []

    with open(filename, 'r') as file:
        line = file.readline().strip()
        assert line=="OFF"

        line = file.readline().strip()
        aux = line.split(' ')

        numVertices = int(aux[0])
        numFaces = int(aux[1])

        for i in range(numVertices):
            aux = file.readline().strip().split(' ')
            vertices += [float(coord) for coord in aux[0:]]
        
        vertices = np.asarray(vertices)
        vertices = np.reshape(vertices, (numVertices, 3))
        print(f'Vertices shape: {vertices.shape}')

        normals = np.zeros((numVertices,3), dtype=np.float32)
        print(f'Normals shape: {normals.shape}')

        for i in range(numFaces):
            aux = file.readline().strip().split(' ')
            aux = [int(index) for index in aux[0:]]
            faces += [aux[1:]]
            
            vecA = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1], vertices[aux[2]][2] - vertices[aux[1]][2]]
            vecB = [vertices[aux[3]][0] - vertices[aux[2]][0], vertices[aux[3]][1] - vertices[aux[2]][1], vertices[aux[3]][2] - vertices[aux[2]][2]]

            res = np.cross(vecA, vecB)
            normals[aux[1]][0] += res[0]  
            normals[aux[1]][1] += res[1]  
            normals[aux[1]][2] += res[2]  

            normals[aux[2]][0] += res[0]  
            normals[aux[2]][1] += res[1]  
            normals[aux[2]][2] += res[2]  

            normals[aux[3]][0] += res[0]  
            normals[aux[3]][1] += res[1]  
            normals[aux[3]][2] += res[2]  
        #print(faces)
        norms = np.linalg.norm(normals,axis=1)
        normals = normals/norms[:,None]

        color = np.asarray(color)
        color = np.tile(color, (numVertices, 1))

        vertexData = np.concatenate((vertices, color), axis=1)
        vertexData = np.concatenate((vertexData, normals), axis=1)

        print(vertexData.shape)

        indices = []
        vertexDataF = []
        index = 0

        for face in faces:
            vertex = vertexData[face[0],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[1],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[2],:]
            vertexDataF += vertex.tolist()
            
            indices += [index, index + 1, index + 2]
            index += 3        



        return bs.Shape(vertexDataF, indices)

def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def createTexturedArc(d):
    vertices = [d, 0.0, 0.0, 0.0, 0.0,
                d+1.0, 0.0, 0.0, 1.0, 0.0]
    
    currentIndex1 = 0
    currentIndex2 = 1

    indices = []

    cont = 1
    cont2 = 1

    for angle in range(4, 185, 5):
        angle = np.radians(angle)
        rot = tr.rotationY(angle)
        p1 = rot.dot(np.array([[d],[0],[0],[1]]))
        p2 = rot.dot(np.array([[d+1],[0],[0],[1]]))

        p1 = np.squeeze(p1)
        p2 = np.squeeze(p2)
        
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])
        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        
        indices.extend([currentIndex1, currentIndex2, currentIndex2+1])
        indices.extend([currentIndex2+1, currentIndex2+2, currentIndex1])

        if cont > 4:
            cont = 0


        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])

        currentIndex1 = currentIndex1 + 4
        currentIndex2 = currentIndex2 + 4
        cont2 = cont2 + 1
        cont = cont + 1

    return bs.Shape(vertices, indices)

def createTiledFloor(dim):
    vert = np.array([[-0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5],[0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0]], np.float32)
    rot = tr.rotationX(-np.pi/2)
    vert = rot.dot(vert)

    indices = [
         0, 1, 2,
         2, 3, 0]

    vertFinal = []
    indexFinal = []
    cont = 0

    for i in range(-dim,dim,1):
        for j in range(-dim,dim,1):
            tra = tr.translate(i,0.0,j)
            newVert = tra.dot(vert)

            v = newVert[:,0][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 1])
            v = newVert[:,1][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 1])
            v = newVert[:,2][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 0])
            v = newVert[:,3][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 0])
            
            ind = [elem + cont for elem in indices]
            indexFinal.extend(ind)
            cont = cont + 4

    return bs.Shape(vertFinal, indexFinal)

# TAREA3: Implementa la función "createHouse" que crea un objeto que representa una casa
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)

def createHouse(pipeline):
    # Quad base para las paredes de la casa

    wallbase = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    wallbase.texture = es.textureSimpleSetup(
        getAssetPath("wall4.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    # Quad base para el techo de la casa

    roofbase = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    roofbase.texture = es.textureSimpleSetup(
        getAssetPath("roof1.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    # Nodos para el grafo de escena

    structureNode = sg.SceneGraphNode('structure')
    roofNode = sg.SceneGraphNode('roof')
    houseNode = sg.SceneGraphNode('house')

    # Nodos para las paredes de la casa

    wallNode1 = sg.SceneGraphNode('wall1')
    wallNode2 = sg.SceneGraphNode('wall2')
    wallNode3 = sg.SceneGraphNode('wall3')
    wallNode4 = sg.SceneGraphNode('wall4')

    wallNode1.transform = tr.matmul([tr.translate(0, 0, 0), tr.uniformScale(1)])
    wallNode1.childs += [wallbase]

    wallNode2.transform = tr.matmul([tr.translate(0, 0, -1), tr.uniformScale(1)])
    wallNode2.childs += [wallbase]

    wallNode3.transform = tr.matmul([tr.rotationY(np.pi/2), tr.translate(0.5, 0.0, -0.5), tr.uniformScale(1)])
    wallNode3.childs += [wallbase]

    wallNode4.transform = tr.matmul([tr.rotationY(np.pi/2), tr.translate(0.5, 0.0, 0.5), tr.uniformScale(1)])
    wallNode4.childs += [wallbase]

    roofNode1 = sg.SceneGraphNode('roof1')
    roofNode2 = sg.SceneGraphNode('roof2')
    roofNode3 = sg.SceneGraphNode('roof3')
    roofNode4 = sg.SceneGraphNode('roof4')
    roofNode5 = sg.SceneGraphNode('roof5')
    roofNode6 = sg.SceneGraphNode('roof6')

    roofNode1.transform = tr.matmul([tr.translate(0.0, 0.5, -0.5), tr.rotationX(np.pi/2), tr.uniformScale(1.2)])
    roofNode1.childs += [roofbase]

    roofNode2.transform = tr.matmul([tr.translate(0.0, 0.6, 0.12), tr.scale(1.2, 0.2, 1.0)])
    roofNode2.childs += [roofbase]

    roofNode3.transform = tr.matmul([tr.translate(0.0, 0.6, -1.1), tr.scale(1.2, 0.2, 1.0)])
    roofNode3.childs += [roofbase]

    roofNode4.transform = tr.matmul([tr.translate(0.6, 0.6, -0.5), tr.scale(1.0, 0.2, 1.0), tr.rotationY(np.pi/2)])
    roofNode4.childs += [roofbase]

    roofNode5.transform = tr.matmul([tr.translate(-0.6, 0.6, -0.5), tr.scale(1.0, 0.2, 1.2), tr.rotationY(np.pi/2)])
    roofNode5.childs += [roofbase]

    roofNode6.transform = tr.matmul([tr.translate(0.0, 0.65, -0.5), tr.rotationX(np.pi/2), tr.uniformScale(1.2)])
    roofNode6.childs += [roofbase]

    structureNode.childs += [wallNode1, wallNode2, wallNode3, wallNode4]
    roofNode.childs += [roofNode1, roofNode2, roofNode3, roofNode4, roofNode5, roofNode6]

    secondRoof = sg.SceneGraphNode('secondroof')
    secondRoof.transform = tr.matmul([tr.translate(0.0, 0.3, 0.0), tr.uniformScale(0.8)])
    secondRoof.childs += [roofNode]

    thirdRoof = sg.SceneGraphNode('thirdRoof')
    thirdRoof.transform = tr.matmul([tr.translate(0.0, 0.55, 0.0), tr.uniformScale(0.6)])
    thirdRoof.childs += [roofNode]

    houseNode.childs += [structureNode, roofNode, secondRoof, thirdRoof]
    return houseNode

# TAREA3: Implementa la función "createWall" que crea un objeto que representa un muro
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)

def createWall(pipeline):
    
    # Quad base para la pared con su textura

    wall = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    wall.texture = es.textureSimpleSetup(
        getAssetPath("wall5.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    # Nodo del grafo de escena que tendra la geometria de la pared y las texturas
    
    wallNode = sg.SceneGraphNode('wall')

    # Nodos para las distintas partes de las paredes
    
    wallNode1 = sg.SceneGraphNode('wall1')
    wallNode2 = sg.SceneGraphNode('wall2')
    wallNode3 = sg.SceneGraphNode('wall3')
    wallNode4 = sg.SceneGraphNode('wall4')
    wallNode5 = sg.SceneGraphNode('wall5')

    wallNode1.transform = tr.matmul([tr.translate(0, 0, 0), tr.uniformScale(1)])
    wallNode1.childs += [wall]

    wallNode2.transform = tr.matmul([tr.translate(0, 0, -0.2), tr.uniformScale(1)])
    wallNode2.childs += [wall]

    wallNode3.transform = tr.matmul([tr.translate(0.5, 0.0, -0.1), tr.scale(1.0, 1.0, 0.2), tr.rotationY(np.pi/2)])
    wallNode3.childs += [wall]

    wallNode4.transform = tr.matmul([tr.translate(-0.5, 0.0, -0.1), tr.scale(1.0, 1.0, 0.2), tr.rotationY(np.pi/2)])
    wallNode4.childs += [wall]

    wallNode5.transform = tr.matmul([tr.translate(0.0, 0.5, 0.0), tr.scale(1.0, 1.0, 0.015), tr.rotationX(np.pi/2)])
    wallNode5.childs += [wall]

    wallNode.childs += [wallNode1, wallNode2, wallNode3, wallNode4, wallNode5]
    return wallNode


# TAREA3: Esta función crea un grafo de escena especial para el auto.
def createCarScene(pipeline):
    chasis = createOFFShape(pipeline, 'alfa2.off', 1.0, 0.0, 0.0)
    wheel = createOFFShape(pipeline, 'wheel.off', 0.0, 0.0, 0.0)

    scale = 2.0
    rotatingWheelNode = sg.SceneGraphNode('rotatingWheel')
    rotatingWheelNode.childs += [wheel]

    chasisNode = sg.SceneGraphNode('chasis')
    chasisNode.transform = tr.uniformScale(scale)
    chasisNode.childs += [chasis]

    wheel1Node = sg.SceneGraphNode('wheel1')
    wheel1Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.056390,0.037409,0.091705)])
    wheel1Node.childs += [rotatingWheelNode]

    wheel2Node = sg.SceneGraphNode('wheel2')
    wheel2Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.060390,0.037409,-0.091705)])
    wheel2Node.childs += [rotatingWheelNode]

    wheel3Node = sg.SceneGraphNode('wheel3')
    wheel3Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.056390,0.037409,0.091705)])
    wheel3Node.childs += [rotatingWheelNode]

    wheel4Node = sg.SceneGraphNode('wheel4')
    wheel4Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.066090,0.037409,-0.091705)])
    wheel4Node.childs += [rotatingWheelNode]

    car1 = sg.SceneGraphNode('car1')
    car1.transform = tr.matmul([tr.translate(2.0, -0.037409, 5.0), tr.rotationY(np.pi)])
    car1.childs += [chasisNode]
    car1.childs += [wheel1Node]
    car1.childs += [wheel2Node]
    car1.childs += [wheel3Node]
    car1.childs += [wheel4Node]

    scene = sg.SceneGraphNode('system')
    scene.childs += [car1]

    return scene

# TAREA3: Esta función crea toda la escena estática y texturada de esta aplicación.
# Por ahora ya están implementadas: la pista y el terreno
# En esta función debes incorporar las casas y muros alrededor de la pista

def createStaticScene(pipeline):

    roadBaseShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    roadBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Road_001_basecolor.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    sandBaseShape = createGPUShape(pipeline, createTiledFloor(50))
    sandBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Sand 002_COLOR.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    arcShape = createGPUShape(pipeline, createTexturedArc(1.5))
    arcShape.texture = roadBaseShape.texture

    paredNode = createWall(pipeline)
    paredNode.transform = tr.matmul([tr.translate(2.0, 0.0, 0.0), tr.rotationY(np.pi/2)])

    houseNode = createHouse(pipeline)
    houseNode.transform = tr.matmul([tr.translate(-4, 0.0, 0.0)])
    
    houseNode2 = createHouse(pipeline)
    houseNode2.transform = tr.matmul([tr.translate(-4, 0.0, 2.0)])

    houseNode3 = createHouse(pipeline)
    houseNode3.transform = tr.matmul([tr.translate(-4, 0.0, -2.0)])

    houseNode4 = createHouse(pipeline)
    houseNode4.transform = tr.matmul([tr.translate(4, 0.0, 0.0)])

    houseNode5 = createHouse(pipeline)
    houseNode5.transform = tr.matmul([tr.translate(4, 0.0, 2.0)])

    houseNode6 = createHouse(pipeline)
    houseNode6.transform = tr.matmul([tr.translate(4, 0.0, -2.0)])

    roadBaseNode = sg.SceneGraphNode('plane')
    roadBaseNode.transform = tr.rotationX(-np.pi/2)
    roadBaseNode.childs += [roadBaseShape]

    arcNode = sg.SceneGraphNode('arc')
    arcNode.childs += [arcShape]

    sandNode = sg.SceneGraphNode('sand')
    sandNode.transform = tr.translate(0.0,-0.1,0.0)
    sandNode.childs += [sandBaseShape]

    linearSector = sg.SceneGraphNode('linearSector')
        
    for i in range(10):
        node = sg.SceneGraphNode('road'+str(i)+'_ls')
        node.transform = tr.translate(0.0,0.0,-1.0*i)
        node.childs += [roadBaseNode]
        linearSector.childs += [node]

    linearSectorLeft = sg.SceneGraphNode('lsLeft')
    linearSectorLeft.transform = tr.translate(-2.0, 0.0, 5.0)
    linearSectorLeft.childs += [linearSector]

    linearSectorRight = sg.SceneGraphNode('lsRight')
    linearSectorRight.transform = tr.translate(2.0, 0.0, 5.0)
    linearSectorRight.childs += [linearSector]

    arcTop = sg.SceneGraphNode('arcTop')
    arcTop.transform = tr.translate(0.0,0.0,-4.5)
    arcTop.childs += [arcNode]

    arcBottom = sg.SceneGraphNode('arcBottom')
    arcBottom.transform = tr.matmul([tr.translate(0.0,0.0,5.5), tr.rotationY(np.pi)])
    arcBottom.childs += [arcNode]
   
    scene = sg.SceneGraphNode('system')
    scene.childs += [linearSectorLeft]
    scene.childs += [linearSectorRight]
    scene.childs += [arcTop]
    scene.childs += [arcBottom]
    scene.childs += [sandNode]
    scene.childs += [paredNode]
    scene.childs += [houseNode, houseNode2, houseNode3, houseNode4, houseNode5, houseNode6]
    
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
    axisPipeline = es.SimpleModelViewProjectionShaderProgram()
    texPipeline = es.SimpleTextureModelViewProjectionShaderProgram()
    lightPipeline = ls.SimpleGouraudShaderProgram()
    
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
    dibujo = createStaticScene(texPipeline)
    car =createCarScene(lightPipeline)

    setPlot(texPipeline, axisPipeline,lightPipeline)

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

        setView(texPipeline, axisPipeline, lightPipeline)

        if controller.showAxis:
            glUseProgram(axisPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            axisPipeline.drawCall(gpuAxis, GL_LINES)

        #NOTA: Aquí dibujas tu objeto de escena
        glUseProgram(texPipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, texPipeline, "model")

        glUseProgram(lightPipeline.shaderProgram)
        sg.drawSceneGraphNode(car, lightPipeline, "model")

        

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()
    

    glfw.terminate()