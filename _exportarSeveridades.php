private function _exportarSeveridades0ou1()
    {
        $arquivo = "/var/www/html/Severidades.xlsx";
        $inputFileType = \PHPExcel_IOFactory::identify($arquivo);
        $objReader     = \PHPExcel_IOFactory::createReader($inputFileType);
        $objReader->setReadDataOnly(true);
        $objPHPExcel = $objReader->load($arquivo);

        $worksheet = $objPHPExcel->getSheet(0);

        $totalLinhas = $worksheet->getHighestRow();

        $severidade0 = '';
        $severidade1 = '';

        $defeito = [];
        for ($linha = 0; $linha <= $totalLinhas; $linha++) {
            $key = trim($worksheet->getCellByColumnAndRow(0, $linha)
                ->getValue());
            $value = trim($worksheet->getCellByColumnAndRow(1, $linha)
                ->getValue());

            if ($key) {
                $defeito[$key] = $value;

                if ($key == 'severity_words') {
                    if ($value) {
                        if (is_numeric($defeito['severity'])) {
                            if ($defeito['severity'] == 1) {
                                $defeito['severity'] = "0";
                            } elseif ($defeito['severity'] == 2) {
                                $defeito['severity'] = "0";
                            } elseif ($defeito['severity'] == 3) {
                                $defeito['severity'] = "1";
                            } elseif ($defeito['severity'] == 4) {
                                $defeito['severity'] = "1";
                            } elseif ($defeito['severity'] == 5) {
                                $defeito['severity'] = "1";
                            }

                            $json = json_encode(
                                $defeito, JSON_UNESCAPED_UNICODE
                            );

                            if ($defeito['severity'] == 0) {
                                $severidade0 .= $json;
                                $severidade0 .= "\n";
                            } elseif ($defeito['severity'] == 1) {
                                $severidade1 .= $json;
                                $severidade1 .= "\n";
                            }
                        }
                    }

                    $defeito = [];
                }
            }
        }

        $file = fopen("/var/www/html/public/severidade/severity1.txt", 'a');
        fwrite($file, $severidade1);
        fclose($file);

        $file = fopen("/var/www/html/public/severidade/severity0.txt", 'a');
        fwrite($file, $severidade0);
        fclose($file);
    }

    private function _exportarSeveridades()
    {
        $arquivo = "/var/www/html/Severidades.xlsx";
        $inputFileType = \PHPExcel_IOFactory::identify($arquivo);
        $objReader     = \PHPExcel_IOFactory::createReader($inputFileType);
        $objReader->setReadDataOnly(true);
        $objPHPExcel = $objReader->load($arquivo);

        $worksheet = $objPHPExcel->getSheet(0);

        $totalLinhas = $worksheet->getHighestRow();

        $severidade1 = '';
        $severidade2 = '';
        $severidade3 = '';
        $severidade4 = '';
        $severidade5 = '';

        $defeito = [];
        for ($linha = 0; $linha <= $totalLinhas; $linha++) {
            $key = trim($worksheet->getCellByColumnAndRow(0, $linha)
                ->getValue());
            $value = trim($worksheet->getCellByColumnAndRow(1, $linha)
                ->getValue());

            if ($key) {
                $defeito[$key] = $value;

                if ($key == 'severity_words') {
                    if ($value) {
                        if (is_numeric($defeito['severity'])) {


                            if ($defeito['severity'] == 1) {
                                $defeito['severity'] = "1";
                                $json = json_encode(
                                    $defeito, JSON_UNESCAPED_UNICODE
                                );

                                $severidade1 .=  $json;
                                $severidade1 .= "\n";
                            } elseif ($defeito['severity'] == 2) {
                                $defeito['severity'] = "2";
                                $json = json_encode(
                                    $defeito, JSON_UNESCAPED_UNICODE
                                );

                                $severidade2 .=  $json;
                                $severidade2 .= "\n";
                            } elseif ($defeito['severity'] == 3) {
                                $defeito['severity'] = "3";
                                $json = json_encode(
                                    $defeito, JSON_UNESCAPED_UNICODE
                                );

                                $severidade3 .=  $json;
                                $severidade3 .= "\n";
                            } elseif ($defeito['severity'] == 4) {
                                $defeito['severity'] = "4";
                                $json = json_encode(
                                    $defeito, JSON_UNESCAPED_UNICODE
                                );

                                $severidade4 .=  $json;
                                $severidade4 .= "\n";
                            } elseif ($defeito['severity'] == 5) {
                                $defeito['severity'] = "5";
                                $json = json_encode(
                                    $defeito, JSON_UNESCAPED_UNICODE
                                );

                                $severidade5 .=  $json;
                                $severidade5 .= "\n";
                            }
                        }
                    }

                    $defeito = [];
                }
            }
        }

        $file = fopen("/var/www/html/public/severidade/severity1.txt", 'a');
        fwrite($file, $severidade1);
        fclose($file);

        $file = fopen("/var/www/html/public/severidade/severity2.txt", 'a');
        fwrite($file, $severidade2);
        fclose($file);

        $file = fopen("/var/www/html/public/severidade/severity3.txt", 'a');
        fwrite($file, $severidade3);
        fclose($file);

        $file = fopen("/var/www/html/public/severidade/severity4.txt", 'a');
        fwrite($file, $severidade4);
        fclose($file);

        $file = fopen("/var/www/html/public/severidade/severity5.txt", 'a');
        fwrite($file, $severidade5);
        fclose($file);
    }