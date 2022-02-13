# 9/22/2021-1
# updated 02/12/2022
from __future__ import absolute_import, division
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, uuid
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
from os.path import *
from time import sleep, perf_counter, perf_counter_ns
from BashColors import C
from TarfileFunctions import *
import tensorflow as tf

class CV2_Utils:
    ''' '''
    __all__ = ['cvu', 'contentPath', 'updated'
                'originalImageZeroPixel', 'zeroPixel']
    
    __builtin__ = [
        'addTwoImages', 'changeBackgroundColors', 'contentPath', 
        'createCV2ImagesTarfile', 'createImageWithColor', 
        'cv2FillImage', 'cvu', 'fillImage', 'fillImage2',
        'fillImage2X', 'flipImage', 'getCV2Image',
        'getNewSavePath', 'getUniqueFileName',
        'edgeDetection', 'plotShowTwoImages', 'printTime',
        'plotShowSingleImage', 'originalImageZeroPixel', 
        'renameFileWithPath', 'resizeImage', 'rotateImage',
        'saveImage', 'showTwoImages', 'translateImage', 'updated',
        'zeroPixel','zoomImage']

    def __init__(self):
        ''' '''
        self.cvu = CV2_Utils
        self.updated = 'updated: 02/12/2022'
        # print(f'{C.BIPurple}{self.updated}{C.ColorOff}')
        self.contentPath = os.getcwd()
        self.zeroPixel = np.array([0,0,0])
        # self.originalImageZeroPixel = np.array([0,0,0])
        self.originalImageZeroPixel = None
        
    def __iter__(self):
        ''' '''
        for key in self.some_sequence:
            yield (key, 'Value for {}'.format(key))
                                               
    def createCV2ImagesTarfile(self):
        ''' '''
        os.chdir(mydrivePth)
        from TarfileFunctions import tff, tarfileFromDirectory
        thisDir = cvu.cv2ImagesPth
        if exists(thisDir):
            # print(thisDir)
            try: tff.tarfileFromDirectory(output_filename='CV2Images.tar.gz',
                                          source_dir=thisDir)
            except BaseException as err: print(err)

    def renameFileWithPath(self, path, append=''):
        '''return newPath'''
        file_path, extension = path.split('.')
        # print(f'file_path: {file_path}')
        # print(f'extension: {extension}')
        newPath = file_path + append + '.' + extension
        newPath = join(newPath)
        
        return newPath
        
    def showTwoImages(self, img1, img2):
        '''display two images with  CV2'''
        img1 = self.resizeImage(img1)
        img2 = self.resizeImage(img2)
        combinedImage = cv2.hconcat([img1, img2])
        try:
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            cv2_imshow(combinedImage)
        except: self.plotShowTwoImages(img1, img2)

    def plotShowSingleImage(self, thisImage, title1='', showAxis=False):
        ''' '''
        thisImage = cv2.cvtColor(thisImage, cv2.COLOR_BGR2RGB)

        fig=plt.figure()
        ax1=fig.add_subplot(1,1,1)
        ax1.imshow(thisImage)
        if showAxis: ax1.axis('on')
        else: ax1.axis('off')
        ax1.set_title(title1)
        plt.show()

    def plotShowTwoImages(self, thisImage, compareImage,
                                title1='Original Image',
                                title2='', showAxis=False):
        ''' '''
        thisImage = cv2.cvtColor(thisImage, cv2.COLOR_BGR2RGB)
        compareImage = cv2.cvtColor(compareImage, cv2.COLOR_BGR2RGB)

        fig=plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(thisImage)
        if showAxis: ax1.axis('on')
        else: ax1.axis('off')
        ax1.set_title(title1)

        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(compareImage)
        if showAxis: ax2.axis('on')
        else: ax2.axis('off')
        ax2.set_title(title2)
        plt.show()

    def resizeImage(self, thisImage, newSize=(224,224)):
        '''return resized_image'''
        resized_image = cv2.resize(thisImage, newSize,
                                   interpolation=cv2.INTER_CUBIC)
        return resized_image

    def createImageWithColor(self, pxColor, silent=True):
        '''return save_path'''
        bgImage = np.zeros(shape=[224,224,3], dtype=np.uint8)
        save_path = join(self.contentPath, 'bgImage.png')

        for px in bgImage:
            bgImage[:] = pxColor
            sleep(0.1)
        cv2.imwrite(save_path, bgImage)
        
        if not silent:
            print(f'shape: {bgImage.shape}')
            bgImage = cv2.imread(save_path, cv2.IMREAD_COLOR)
            # bgImage = cv2.cvtColor(bgImage, cv2.COLOR_BGR2RGB)
            try:
                cv2_imshow(bgImage)
                cv2.waitKey(100)
                cv2.destroyAllWindows()
            except:
                self.plotShowSingleImage(bgImage, title1=basename(save_path))
        return save_path

    def zoomImage(self, thisImage, scale=1, silent=True):
        '''return zoomImage'''
        zeroPixel = thisImage[0][0]
        angle = 0
        width, height, _ = thisImage.shape
        rotPoint = width//2, height //2
        dimentions = width, height
        rotPoint = width//2, height//2
        rotMatrix = cv2.getRotationMatrix2D(
            rotPoint, angle, scale=scale)
        zoomImage = cv2.warpAffine(thisImage, rotMatrix, dimentions)
        # zoomImage = self.cv2FillImage(zoomImage)
        if not silent:
            print('cv2ZoomImage()')
            self.plotShowTwoImages(thisImage, zoomImage)
        return zoomImage

    def rotateImage(self, thisImage,
                    angle=0, rotPoint=None, scale=1, silent=True):
        '''return rotImage'''
        thisImage = np.copy(thisImage)
        zeroPixel = thisImage[[0][0]]
        width, height, _ = thisImage.shape
        if rotPoint == None:
            rotPoint = width//2, height//2
            rotMat = cv2.getRotationMatrix2D(rotPoint, angle, scale=scale)
            dimentions = width, height
            rotImage = cv2.warpAffine(thisImage, rotMat, dimentions)
            rotImageZeroPixel = rotImage[0][0]
            if not silent:
                print('cv2Rotation()')
                self.plotShowTwoImages(thisImage, rotImage)
                sleep(0.1)
            try:
                cv2.waitKey(100)
                cv2.destroyAllWindows()
            except: pass
            return rotImage

    def translateImage(self, thisImage, x=0, y=0, silent=True):
        '''-x shift left -y shift up\nx shift right y shift down\n
        return newImage'''
        thisImage = np.copy(thisImage)
        zeroPixel=thisImage[0][0]
        translateMatrix = np.float32([[1,0,x],[0,1,y]])
        dimentions = (thisImage.shape[1], thisImage.shape[0])
        newImage = cv2.warpAffine(thisImage, translateMatrix, dimentions)

        if not silent:
            print('cv2Translate()')
            self.plotShowTwoImages(thisImage, newImage)

        try:
            cv2.waitKey(100)
            cv2.destroyAllWindows()
        except: pass
        return newImage

    def edgeDetection(self, thisImage, t1=50, t2=100, silent=True):
        '''return canny'''

        # gray = cv2.cvtColor(thisImage, cv2.COLOR_BGR2GRAY)
        # thisImage = cv2.GaussianBlur(gray,(3,3),0)

        canny = cv2.Canny(thisImage, t1, t2)
        if not silent:
            self.plotShowTwoImages(thisImage, canny)
        try:
            cv2.waitKey(100)
            cv2.destroyAllWindows()
        except: pass
        return canny

    def changeBackgroundColors(self, path, image=None):
        '''returns newImg'''
        splitPath = split(path)
        filePath = splitPath[0]
        fileName = splitPath[1]
        fileName = 'new_' + fileName
        print(filePath)
        print(fileName)
        saveImagePath = join(contentPath, fileName)

        print(f'saveImagePath: {saveImagePath}')
        original_image = cv2.imread(path, cv2.IMREAD_COLOR)
        originalZeroPixel = original_image[0][0]
        plt.imshow(original_image)
        newImg = np.copy(original_image)

        zeroPixel = newImg[0][0]
        print('newImg zeroPixel:', zeroPixel)
        print('originalZeroPixel:', originalZeroPixel)

        width, height, channels = original_image.shape
        print(width, height, channels)

        for x in range(0, width):
            for y in range(0, height):
                channels_xy = newImg[y][x]
                # print(channels_xy)
                if all(channels_xy == zeroPixel):
                    newImg[y][x] = originalZeroPixel

        cv2.imwrite(saveImagePath, newImg)
        newImg = cv2.imread(saveImagePath, -1)
        try:
            cv2.waitKey(100)
            cv2.destroyAllWindows()
        except: pass
        return newImg

    def cv2FillImage(self, thisImage, silent=True):
        '''returns filledImage'''
        print('cv2FillImage')
        # plotShowSingleImage(thisImage)
        zp = thisImage[0][0]
        bgImagePath = self.cv2CreateImageWithColor(pxColor=cvu.zeroPixel)
        # print(zp)
        img1 = np.copy(thisImage)
        img2 = cv2.imread(bgImagePath, cv2.IMREAD_COLOR)
        filledImage = cv2.bitwise_or(img1, img2)
        filledImage = cv2.bitwise_or(filledImage, img2)
        if not silent:
            self.cv2ShowTwoImages(img1, filledImage)
        try:
            cv2.waitKey(100)
            cv2.destroyAllWindows()
        except: pass
        return filledImage

    def addTwoImages(self, imagePath1, imagePath2, alfa=1, beta=1, gamma=0.0):
        '''return  addImage'''
        # read two imagePaths
        src1 = cv2.imread(imagePath1)
        src2 = cv2.imread(imagePath2)
        try:
            # add or blend the imagePaths
            addImage = cv2.addWeighted(src1, alfa, src2, beta, gamma)
            # save the output imagePath
            # cv2.imwrite('image.png', dst)
            return addImage
        except Exception as err:
            print(f'{C.IRed}{err}')

    def flipImage(self, thisImage, axes=0, silent=True):
        '''axes=0 flip vert\naxes=1 flip horiz\naxes=-1 flip vert and horiz'''
        if axes == 0:
            flipImg = cv2.flip(thisImage, 0)
            if not silent:
                print('_flipVert')
                self.showTwoImages(thisImage, flipImg)
        elif axes == 1:
            flipImg = cv2.flip(thisImage, 1)
            if not silent:
                print('_flipHorz')
                self.showTwoImages(thisImage, flipImg)
        elif axes == -1:
            flipImg = cv2.flip(thisImage, -1)
            if not silent:
                print('_flipVertAndHorz')
                self.showTwoImages(thisImage, flipImg)
        try:
            cv2.waitKey(100)
            cv2.destroyAllWindows()
        except: sleep(0.1)
        return flipImg

    def printTime(self, strt, fin):
        '''print execution times '''
        totalTime = fin - strt
        totalTime= round(totalTime, 0)
        if totalTime < 60.0:
            print(f'completed: {totalTime} second(s)')
        elif totalTime >= 60.0:
            minutes = totalTime//60
            seconds = totalTime - (minutes * 60)
            seconds = round(seconds, 2)
            print(
                f'completed: {minutes} minutes {seconds} second(s)')

    def getCV2Image(self, imagePath:str):
        img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        return img

    def getUniqueFileName(self, uniquePrefix='_'):
        '''return uniqueName'''
        uniqueName = uniquePrefix + str(uuid.uuid4())
        return uniqueName
        
    def getNewSavePath(self, initialPath, new_path='', postfix='', silent=True):
        ''' '''
        if new_path=='':
            new_path = self.cv2ImagesPth + '/'
        
        directory, fileName = split(initialPath)
        # dirr, name, extension = splitext(directory)
        # print(initialPath)
        # print(directory)
        # print(fileName)
        _, subDir = split(directory)
        name, extension = splitext(fileName)
        # print(subDir)

        if not silent:
            print(f'{C.IWhite}')
            print(f'newPath: {new_path}')
            print(f'subDir: {subDir}')
            print(f'name: {name}')
            print(f'postfix: {postfix}')
            print(f'extension: {extension}')

        newSavePath = new_path + subDir + '/' + name + postfix + extension
        print(f'{C.IWhite}savePath: {C.IPurple}{newSavePath}{C.ColorOff}')
        return newSavePath

    def saveImage(self, savePath, thisImage, silent=True, save=False):
        ''' '''
        if not silent:
            print(f'saved to: {C.BIGreen}{savePath}{C.ColorOff}')
            try: cv2_imshow(thisImage)
            except BaseException as err:
                self.plotShowSingleImage(thisImage,
                                         title=basename(savePath))

        if save and not exists(savePath):
            print(f'{C.IWhite}saved to: {C.Green}{savePath}{C.ColorOff}')
            self.cv2ImagePathList.append(savePath)
            cv2.imwrite(savePath, thisImage)

    def fillImage(self, thisImage, silent=True):
        ''' '''
        original_image = np.copy(thisImage)
        img = np.copy(original_image)

        black = np.where((img[:,:,0]==0) & 
                         (img[:,:,1]==0) & 
                         (img[:,:,2]==0))
        
        white = np.where((img[:,:,0]==255) & 
                         (img[:,:,1]==255) & 
                         (img[:,:,2]==255))
        
        img[black] = (255, 255, 255)
        img[white] = (0, 0, 0)
        
        if not silent:
            try:
                cvu.showTwoImages(original_image, img)
            except Exception as err:
                print(err)

        try:
            cv2.waitKey(100)
            cv2.destroyAllWindows()
        except: sleep(0.1)
        return img
    
    def fillImage2(self, thisImage, silent=True):
        ''' '''
        original_image = np.copy(thisImage)
        img = np.copy(original_image)

        black = np.where((img[:,:,0]==0) &
                         (img[:,:,1]==0) &
                         (img[:,:,2]==0))

        white = np.where((img[:,:,0]==255) &
                         (img[:,:,1]==255) &
                         (img[:,:,2]==255))

        img[black] = (255, 255, 255)
        img[white] = (0, 0, 0)

        if not silent:
            cvu.showTwoImages(original_image, img)

    def fillImage2X(self, thisImage, silent=True):
        ''' '''
        imageCopy = np.copy(thisImage)
        intermediateImage = self.fillImage(imageCopy)
        cleanImage = self.fillImage(intermediateImage)
        if not silent:
            print('fillImage2X()')
            edges = self.edgeDetection(cleanImage)
            self.plotShowTwoImages(cleanImage, edges,
                                  title1='cleaned image',
                                  title2='edges')
        return cleanImage
cvu = CV2_Utils()
