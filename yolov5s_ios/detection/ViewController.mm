//
//  ViewController.m
//  detection
//
//  Created by yang on 2020/12/3.
//


#import <opencv2/core/core.hpp>
#import <opencv2/opencv.hpp>
#import <AVFoundation/AVFoundation.h>

#import "bbox.hpp"
#import "inference.hpp"
#import "preprocess.hpp"
#import "postprocess.hpp"
#import "ViewController.h"

@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>{
    ISession *modelSession;
}

@property (strong, nonatomic) IBOutlet UIImageView *imageView;
@property (strong, nonatomic) AVCaptureSession *cameraSession;
@property (strong, nonatomic, readwrite) AVCaptureConnection *videoConnection;

@end

@implementation ViewController


- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"yolov5s_fp16.onnx" ofType:@"mnn"];
    
    InitParam modelParam;
    modelParam.num_threads = 4;
    modelParam.forward_type = kForwardMetal;
    modelParam.precision_type = kFp16;
    
    modelSession = CreateSession(modelPath.UTF8String, modelParam);
    [self.cameraSession startRunning];
}


- (void)awakeFromNib {
    [super awakeFromNib];

    self.cameraSession               = [[AVCaptureSession alloc] init];
    self.cameraSession.sessionPreset = AVCaptureSessionPresetPhoto;
    AVCaptureDevice *device          = [self cameraWithPosition:AVCaptureDevicePositionBack];
    AVCaptureDeviceInput *input      = [[AVCaptureDeviceInput alloc] initWithDevice:device error:NULL];
    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    [output setSampleBufferDelegate:self queue:dispatch_queue_create("huya.detection", 0)];
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};

    if ([self.cameraSession canAddInput:input]) {
        [self.cameraSession addInput:input];
    }
    if ([self.cameraSession canAddOutput:output]) {
        [self.cameraSession addOutput:output];
    }
    [self.cameraSession commitConfiguration];
    
    self.videoConnection =  [output connectionWithMediaType:AVMediaTypeVideo];
    [self.videoConnection setVideoOrientation:AVCaptureVideoOrientationPortrait];
    
    if (device.position == AVCaptureDevicePositionFront)
        [self.videoConnection setVideoMirrored:YES];
}


- (AVCaptureDevice *)cameraWithPosition:(AVCaptureDevicePosition)position {
    AVCaptureDeviceDiscoverySession *devicesIOS10 = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
    
    NSArray *devices = devicesIOS10.devices;
    for ( AVCaptureDevice *device in devices )
        if ( device.position == position ) {
            return device;
        }
    return nil;
}

- (cv::Mat)cvMatFromUIImage:(UIImage *)image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat_c4(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    CGContextRef contextRef = CGBitmapContextCreate(cvMat_c4.data,                  // Pointer to  data
                                                    cols,                           // Width of bitmap
                                                    rows,                           // Height of bitmap
                                                    8,                              // Bits per component
                                                    cvMat_c4.step[0],               // Bytes per row
                                                    colorSpace,                     // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault);     // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    cv::Mat cvMat_c3(rows, cols, CV_8UC3); // 8 bits per component, 3 channels (color channels)
    cv::cvtColor(cvMat_c4, cvMat_c3, cv::COLOR_RGBA2RGB);
    return cvMat_c3;
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat {
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 // width
                                        cvMat.rows,                                 // height
                                        8,                                          // bits per component
                                        8 * cvMat.elemSize(),                       // bits per pixel
                                        cvMat.step[0],                              // bytesPerRow
                                        colorSpace,                                 // colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   // CGDataProviderRef
                                        NULL,                                       // decode
                                        false,                                      // should interpolate
                                        kCGRenderingIntentDefault                   // intent
                                        );
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

#pragma mark AVCaptureAudioDataOutputSampleBufferDelegate
- (void)captureOutput:(AVCaptureOutput *)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
    
    CIImage *ci        = [[CIImage alloc] initWithCVPixelBuffer:CMSampleBufferGetImageBuffer(sampleBuffer)];
    CIContext *context = [[CIContext alloc] init];
    CGImageRef cg      = [context createCGImage:ci fromRect:ci.extent];
    
    UIImage *image = [UIImage imageWithCGImage:cg scale:1.f orientation:UIImageOrientationUp];
    CGImageRelease(cg);
    auto cvImage = [self cvMatFromUIImage:image];
    
    auto data = (float*)malloc(1*12*208*208*sizeof(float));
    preprocess(cvImage, data);
    
    NSTimeInterval begin = NSDate.timeIntervalSinceReferenceDate;
    auto feature_map_ptrs = modelSession->run(data);
    NSTimeInterval cost = NSDate.timeIntervalSinceReferenceDate - begin;
    NSString *time = [@"" stringByAppendingFormat:@"%.3f ms", cost * 1000.f];
    NSLog(@"Inference time elapse: %@", time);
    
    free(data);
    
    auto boxes = postprocess(feature_map_ptrs, 0.3, 0.5);
    scaleCoords(boxes, cvImage.cols, cvImage.rows);
    drawBoxes(cvImage, boxes);
    
    image = [self UIImageFromCVMat:cvImage];
    dispatch_sync(dispatch_get_main_queue(), ^{
        self.imageView.image = image;
    });
}

@end
