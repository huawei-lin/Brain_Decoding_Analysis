<?php

header("Content-Type:text/html;charset=UTF-8");
if(is_uploaded_file($_FILES['file']['tmp_name'])){
    move_uploaded_file($_FILES['file']['tmp_name'], "../drag_uploads/data.zip");
}
$ret = exec("python /var/www/drag_uploads/fMRI/Network.py");
// echo $ret;

$file_path = "/var/www/drag_uploads/fMRI/result.re";
$i = 1;
while ($i <= 10):
    if(file_exists($file_path)){
        $str = file_get_contents($file_path);//将整个文件内容读入到一个字符串中
        $str = str_replace("\n", "", $str); 
        break;
    }
endwhile;

echo $str;

 ?>
