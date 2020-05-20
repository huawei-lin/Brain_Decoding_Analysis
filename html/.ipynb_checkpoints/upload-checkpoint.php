<?php

header("Content-Type:text/html;charset=UTF-8");
if(is_uploaded_file($_FILES['file']['tmp_name'])){
    move_uploaded_file($_FILES['file']['tmp_name'], "../drag_uploads/data.zip");
}
$ret = exec("python /var/www/drag_uploads/four_classify/load.py");
echo $ret;

 ?>
