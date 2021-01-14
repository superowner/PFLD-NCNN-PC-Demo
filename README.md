# PFLD-NCNN-PC-Demo

ref:https://github.com/abyssss52/PFLD_ncnn_test  
ref:https://www.cnblogs.com/chucklu/p/10487659.html  

基本参考了上面的c艹部分代码  
玉兰：https://wx1.sinaimg.cn/mw1024/007Bgb15ly1gmnnh9wh6qj30zi0ikdhu.jpg  
ncnn用的2020年4月或以前的版本，未来版本应该也可以  
OpenCV：410或以上  
vulkan：1.2.162.1  
输入图片为112x112(必须)的普通彩色图片，作为Tensor的输入，可以用centerface，mtcnn，Ultraface等框架提取切割脸部box然后输入pfld网络！  
  
首先安装vulkan，然后编译ncnn，指定vulkan路径编译以支持gpu，ncnn用的gpu指的就是vulkan的gpu加速框架  
然后编译这个，vs引用(include,lib)需要同时引用opencv、vulkan、ncnn的，缺一不可！  
  
 ```cpp
#include <stdio.h>
//#include <time.h>

//CPP
#include <vector>
#include <string>

// calculate euler angle
//#include <opencv.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

////https://blog.csdn.net/jacke121/article/details/101035833
// ncnn
#include "net.h"
#include "benchmark.h"
#include "retina_face.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN




static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net pfld;

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

int SystemInit()
{
#if NCNN_VULKAN
    return ncnn::create_gpu_instance()==0;
#else 
    return -1;
#endif
}
int SystemDestroy()
{
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
    return 0;
}

int main()
{
    auto initRes = SystemInit();
    printf("initRes:%d", initRes);

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    bool use_gpu = false;
    if (1 == initRes)
    {
        // use vulkan compute
        if (ncnn::get_gpu_count() != 0)
        {
            opt.use_vulkan_compute = true;
            use_gpu = true;
        }
    }
    printf("Vulkan Ready ? %s\n",(true==use_gpu)?"Yes":"No");
    pfld.opt = opt;

    int ret = pfld.load_param("./assets/pfld-sim.param");
    if (ret != 0)
    {
        printf("load_param_bin failed\n");
        goto exit_flag;
    }
    ret = pfld.load_model("./assets/pfld-sim.bin");
    if (ret != 0)
    {
        printf("load_model failed\n");
        goto exit_flag;
    }
    printf("init success!\n");
    {
        double start_time = ncnn::get_current_time();
        auto testImg = cv::imread("001.png",1);
        if (testImg.empty())
            return SystemDestroy();
        cv::Size s = testImg.size();
        auto h = s.height;
        auto w = s.width;
        int height = testImg.rows;
        int width = testImg.cols;
        printf("Size of image (h, w) = (%d, %d),%d,%d\n", h, w, height, width);
        if (width != 112 || height != 112)
            return SystemDestroy();
        ncnn::Mat in = ncnn::Mat::from_pixels(testImg.data, ncnn::Mat::PIXEL_BGR2RGB, width, height);

        printf("flag1\n");
        // pfld
        std::vector<float> keypoints;
        {
            //        const float mean_vals[3] = {103.f, 117.f, 123.f};
            const float norm_vals[3] = { 1 / 255.f,1 / 255.f,1 / 255.f };
            in.substract_mean_normalize(0, norm_vals);

            ncnn::Extractor ex = pfld.create_extractor();

            if(use_gpu)
                ex.set_vulkan_compute(use_gpu);

            printf("flag2\n");
            ex.input("input_1", in);

            ncnn::Mat out;
            ex.extract("415", out);
            printf("flag3,%d\n-----------------------\n", out.w);
            keypoints.resize(out.w);
            for (int j = 0; j < out.w; j++)
            {
                keypoints[j] = out[j] * 112;
                //printf("%f\n", out[j] * 112);
                //keypoints.push_back(out[j] * 112);
            }
            size_t max_len = keypoints.size();
            printf("max_len:%d\n",max_len);
            printf("flag5\n");
            for (size_t i = 0; i < max_len; i+=2)
            {
                cv::circle(testImg,cv::Point(keypoints[i], keypoints[i+1]),2,cv::Scalar(1,0,0));
            }
            cv::imshow("abc",testImg);
            cv::waitKey();
        }
    }
    printf("finish\n");
    exit_flag:;
    SystemDestroy();
    getchar();
    return 0;
}
 ```
