private function _severidadesToExcel()
    {
        $arquivos = $this->moveFilesToTempFolder();

        if (count($arquivos) == 0) {
            throw new DSINException(
                'Planilha não encontrada. Verifique!!!'
            );
        }

        foreach ($arquivos as $arquivo) {
            $arquivo = TMP_DIR . $arquivo;

            $file = fopen($arquivo, "r");

            $mode = '';
            $classifier = '';
            $metric = '';
            $results = [];
            while (!feof($file)) {
                $line = fgets($file);

                $line = str_replace(':', '', $line);
                $line = str_replace('{', '', $line);
                $line = str_replace('}', '', $line);
                $line = str_replace('"', '', $line);
                $line = str_replace('\n', '', $line);
                $line = str_replace('\'', '', $line);
                $line = str_replace(',', '', $line);
                $line = str_replace('array([', '', $line);

                if (strpos($line, 'Mode') !== false) {
                    $line = str_replace('Mode', '', $line);
                    $line = trim($line);
                    $mode = $line;
                    $results[$mode] = [];
                }

                if (strpos($line, 'Classifier') !== false) {
                    $line = str_replace('Classifier', '', $line);
                    $line = trim($line);
                    $classifier = $line;
                    $results[$mode][$classifier] = [];
                }

                if (strpos($line, 'Accuracy') !== false) {
                    $metric = 'Accuracy';
                    $line = str_replace('Accuracy', '', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric] = [];
                }

                if (strpos($line, 'F1') !== false) {
                    $metric = 'F1';
                    $line = str_replace('F1', '', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric] = [];
                }

                if (strpos($line, 'Precision') !== false) {
                    $metric = 'Precision';
                    $line = str_replace('Precision', '', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric] = [];
                }

                if (strpos($line, 'Recall') !== false) {
                    $metric = 'Recall';
                    $line = str_replace('Recall', '', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric] = [];
                }


                if (strpos($line, 'avg') !== false) {
                    $line = str_replace('avg', '', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric]['avg'] = $line;
                }

                if (strpos($line, 'max') !== false) {
                    $line = str_replace('max', '', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric]['max'] = $line;
                }

                if (strpos($line, 'median') !== false) {
                    $line = str_replace('median', '', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric]['median'] = $line;
                }

                if (strpos($line, 'min') !== false) {
                    $line = str_replace('min', '', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric]['min'] = $line;
                }

                if (strpos($line, 'std') !== false) {
                    $line = str_replace('std', '', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric]['std'] = $line;
                }

                if (strpos($line, 'values') !== false) {
                    $line = str_replace('values', '', $line);
                    $line = str_replace('  ', ' ', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric]['values']
                        = explode(' ', $line);
                }

                if (strpos($line, '])') !== false) {
                    $line = str_replace('])', '', $line);
                    $line = str_replace('  ', ' ', $line);
                    $line = trim($line);
                    $results[$mode][$classifier][$metric]['values'] =
                    array_merge(
                        $results[$mode][$classifier][$metric]['values'],
                        explode(' ', $line)
                    );
                }


            }
            fclose($file);

            $objPHPExcel = new \PHPExcel();
            $objPHPExcel->setActiveSheetIndex(0);

            $sheet =  $objPHPExcel->getActiveSheet();

            $sheet->setCellValueByColumnAndRow(0,1, 'Metric');

            $sheet->setCellValueByColumnAndRow(2,1, '2 Class');

            $sheet->setCellValueByColumnAndRow(2,2, 'Default');
            $sheet->setCellValueByColumnAndRow(10,2, 'Domain');
            $sheet->setCellValueByColumnAndRow(18,2, 'Uses');
            $sheet->setCellValueByColumnAndRow(26,2, 'Alter Uses');


            $sheet->setCellValueByColumnAndRow(2,3, 'LogisticRegression');
            $sheet->setCellValueByColumnAndRow(3,3, 'MultinomialNB');
            $sheet->setCellValueByColumnAndRow(4,3, 'AdaBoost');
            $sheet->setCellValueByColumnAndRow(5,3, 'SVC');
            $sheet->setCellValueByColumnAndRow(6,3, 'LinearSVC');
            $sheet->setCellValueByColumnAndRow(7,3, 'SVCScale');
            $sheet->setCellValueByColumnAndRow(8,3, 'DecisionTree');
            $sheet->setCellValueByColumnAndRow(9,3, 'RandomForest');

            $sheet->setCellValueByColumnAndRow(10,3, 'LogisticRegression');
            $sheet->setCellValueByColumnAndRow(11,3, 'MultinomialNB');
            $sheet->setCellValueByColumnAndRow(12,3, 'AdaBoost');
            $sheet->setCellValueByColumnAndRow(13,3, 'SVC');
            $sheet->setCellValueByColumnAndRow(14,3, 'LinearSVC');
            $sheet->setCellValueByColumnAndRow(15,3, 'SVCScale');
            $sheet->setCellValueByColumnAndRow(16,3, 'DecisionTree');
            $sheet->setCellValueByColumnAndRow(17,3, 'RandomForest');


            $sheet->setCellValueByColumnAndRow(18,3, 'LogisticRegression');
            $sheet->setCellValueByColumnAndRow(19,3, 'MultinomialNB');
            $sheet->setCellValueByColumnAndRow(20,3, 'AdaBoost');
            $sheet->setCellValueByColumnAndRow(21,3, 'SVC');
            $sheet->setCellValueByColumnAndRow(22,3, 'LinearSVC');
            $sheet->setCellValueByColumnAndRow(23,3, 'SVCScale');
            $sheet->setCellValueByColumnAndRow(24,3, 'DecisionTree');
            $sheet->setCellValueByColumnAndRow(25,3, 'RandomForest');

            $sheet->setCellValueByColumnAndRow(26,3, 'LogisticRegression');
            $sheet->setCellValueByColumnAndRow(27,3, 'MultinomialNB');
            $sheet->setCellValueByColumnAndRow(28,3, 'AdaBoost');
            $sheet->setCellValueByColumnAndRow(29,3, 'SVC');
            $sheet->setCellValueByColumnAndRow(30,3, 'LinearSVC');
            $sheet->setCellValueByColumnAndRow(31,3, 'SVCScale');
            $sheet->setCellValueByColumnAndRow(32,3, 'DecisionTree');
            $sheet->setCellValueByColumnAndRow(33,3, 'RandomForest');

            $pos_approach['Default'] = 2;
            $pos_approach['Domain'] = 10;
            $pos_approach['Uses'] = 18;
            $pos_approach['Alter Uses'] = 26;

            $pos_model['LogisticRegression'] = 0;
            $pos_model['MultinomialNB'] = 1;
            $pos_model['AdaBoost'] = 2;
            $pos_model['SVC'] = 3;
            $pos_model['LinearSVC'] = 4;
            $pos_model['SVCScale'] = 5;
            $pos_model['DecisionTree'] = 6;
            $pos_model['RandomForest'] = 7;



            $pos_metric['F1'] = 4;
            $pos_metric['Accuracy'] = 19;
            $pos_metric['Precision'] = 34;
            $pos_metric['Recall'] = 49;

            $pos_val['1'] = 0;
            $pos_val['2'] = 1;
            $pos_val['3'] = 2;
            $pos_val['4'] = 3;
            $pos_val['5'] = 4;
            $pos_val['6'] = 5;
            $pos_val['7'] = 6;
            $pos_val['8'] = 7;
            $pos_val['9'] = 8;
            $pos_val['10'] = 9;
            $pos_val['avg'] = 10;
            $pos_val['max'] = 11;
            $pos_val['min'] = 12;
            $pos_val['median'] = 13;
            $pos_val['std'] = 14;



            foreach ($results as $approach => $values) {
                foreach ($values as $classifier => $metrics) {
                    foreach ($metrics as $metric => $result) {
                        foreach ($result as $key => $val) {
                            $col = $pos_approach[$approach] + $pos_model[$classifier];

                            if (!is_array($val)) {
                                $row = $pos_metric[$metric] + $pos_val[$key];

                                $sheet->setCellValueByColumnAndRow(
                                    $col,
                                    $row,
                                    $val
                                );
                            } else {
                                $i = 1;
                                foreach ($val as $v) {
                                    $row = $pos_metric[$metric] + $pos_val[$i];
                                    $i++;

                                    $sheet->setCellValueByColumnAndRow(
                                        $col,
                                        $row,
                                        $v
                                    );
                                }
                            }
                        }
                    }
                }
            }

            header("Pragma: public");
            header("Expires: 0");
            header("Cache-Control: must-revalidate, post-check=0, pre-check=0");
            header("Content-Type: application/force-download");
            header("Content-Type: application/octet-stream");
            header("Content-Type: application/download");;
            header("Content-Disposition: attachment;filename=list.xls");
            header("Content-Transfer-Encoding: binary ");

            $objWriter = new \PHPExcel_Writer_Excel2007($objPHPExcel);
            $objWriter->save('/var/www/html/public/severidade/testa.xlsx');


          die();
        }
    }