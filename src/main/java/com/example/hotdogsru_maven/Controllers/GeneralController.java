package com.example.hotdogsru_maven.Controllers;

import com.example.hotdogsru_maven.Service.NeiroModel;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.view.RedirectView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

@Controller
@RequestMapping("/")
public class GeneralController {
    NeiroModel neiroModel;
    String path = "/tmp/server/HotDogs/loaded.jpg";

    public GeneralController() throws IOException, ai.onnxruntime.OrtException {
        neiroModel = new NeiroModel();
        new File("/tmp/server/HotDogs").mkdirs();
        System.out.println("Default picture path: " + path);
        System.out.println("Testing model...");
        try {
            String str1 = neiroModel.getAnswer("src/main/resources/images/1.jpg");
            System.out.println("1.jpg: " + str1);
            String str2 = neiroModel.getAnswer("src/main/resources/static/img/beagle.jpg");
            System.out.println("beagle.jpg: " + str2);
            String str3 = neiroModel.getAnswer("src/main/resources/static/img/korgi.jpg");
            System.out.println("korgi.jpg: " + str3);
            if("Beagle".equals(str1)){
                System.out.println("Test Done!");
            }
            else{
                System.out.println("Test Fail! Got: " + str1);
            }
        } catch (Exception e) {
            System.out.println("Test failed: " + e.getMessage());
        }
    }

//    @GetMapping(produces = MediaType.TEXT_HTML_VALUE)
    @GetMapping("/")
    public String sendIndexPage() {
        System.out.println("send index.html");
        return "index.html";
    }

    @GetMapping(value = "/result", produces = MediaType.TEXT_HTML_VALUE)
    public String sendResultPage(){
        return "result.html";
    }

//    @RequestMapping(value = "/", produces = MediaType.TEXT_HTML_VALUE)
//    public String index2(){
//        System.out.println("send index.html");
//        return "index.html";
//    }

    @GetMapping(value = "/ErrorFile", produces = MediaType.TEXT_HTML_VALUE)
    public String sendErrorFilePage(){
        return "errorDownloadFile.html";
    }

//    @GetMapping(value = "/index.html", produces = MediaType.TEXT_HTML_VALUE)
//    public String index(){
//        System.out.println("send index.html");
//        return "index.html";
//    }

    @GetMapping(value = "/ErrorNoFile", produces = MediaType.TEXT_HTML_VALUE)
    public String sendErrorNoFilePage(){
        return "errorNoFile.html";
    }

    @GetMapping(value = "/error", produces = MediaType.TEXT_HTML_VALUE)
    public String sendErrorPage(){
        return "errorDownloadFile.html";
    }

    @RequestMapping(value = "/uploading", method = RequestMethod.POST)
    public RedirectView provideInfo(@RequestParam("file") MultipartFile file) throws FileNotFoundException {
        if(!file.isEmpty()){
            try {
                file.transferTo(new File(path));
                System.out.println("File loaded "+file.getOriginalFilename());

                String str = neiroModel.getAnswer(path);
                System.out.println("Prediction: picture is " + str);
                System.out.println("Redirect to result.html and add arg="+str);
                return new RedirectView("/result?arg="+str);
            } catch (Exception e){
                System.out.println(e);
                System.out.println("Redirect to ErrorFile.html");
                return new RedirectView("/ErrorFile");
            }
        } else {
            System.out.println("Redirect to ErrorNoFile.html");
            return new RedirectView("/ErrorNoFile");
        }
    }
}