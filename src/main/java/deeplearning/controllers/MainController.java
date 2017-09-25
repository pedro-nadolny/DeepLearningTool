package deeplearning.controllers;

import com.jfoenix.controls.*;

import deeplearning.models.HistoryRow;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.OneR;
import weka.core.Attribute;
import weka.core.DictionaryBuilder;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Array;
import java.net.URL;
import java.util.*;


public class MainController extends VBox implements Initializable{

    @FXML private Label baseInfoAttributesLabel;
    @FXML private Label baseInfoNameLabel;
    @FXML private Label baseInfoClassesLabel;
    @FXML private Label baseInfoInstancesLabel;

    @FXML private JFXComboBox baseClassComboBox;
    @FXML private JFXListView<HistoryRow> historyListView;

    @FXML private JFXRadioButton radioButtonNeuralNetwork;
    @FXML private JFXRadioButton radioButtonRuleBased;
    @FXML private JFXRadioButton radioButtonNaiveBayes;
    @FXML private JFXRadioButton radioButtonC45;

    @FXML private JFXCheckBox checkBoxEnableDL;
    @FXML private JFXCheckBox checkBoxEnableCC;

    @FXML private StackPane root;

    @FXML private VBox parametersBox;

    @FXML private JFXSlider sliderConfidenceFactorC45;
    @FXML private JFXSlider sliderMinLeaftC45;

    @FXML private JFXSlider sliderLearningRateDL;
    @FXML private JFXSlider sliderKernelXDL;
    @FXML private JFXSlider sliderKernelYDL;
    @FXML private JFXSlider sliderStrideXDL;
    @FXML private JFXSlider sliderStrideYDL;
    @FXML private JFXTextField textFieldHiddenLayersDL;

    private JFXSlider sliderLearningrRateNN;
    private JFXSlider sliderMomentumNN;
    private JFXSlider slideValidationSetSizeNN;
    private JFXTextField textFieldHiddenLayersNN;
    
    private JFXSlider sliderMinBucketSizeRB;

    private Instances base;
    private ObservableList<HistoryRow> historyList;

    private Reorder attOrderFilter = new Reorder();

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        sliderLearningrRateNN = new JFXSlider();
        sliderLearningrRateNN.setValue(30);

        sliderMomentumNN = new JFXSlider();
        sliderMomentumNN.setValue(20);
        slideValidationSetSizeNN = new JFXSlider();

        textFieldHiddenLayersNN = new JFXTextField();
        textFieldHiddenLayersNN.setUnFocusColor(Color.WHITE);
        textFieldHiddenLayersNN.setText("a");

        sliderMinBucketSizeRB = new JFXSlider();
        sliderMinBucketSizeRB.setMax(10);
        sliderMinBucketSizeRB.setValue(6);

        historyList = FXCollections.observableArrayList(HistoryRow.extractor());
        historyListView.setItems(historyList);

        historyListView.setCellFactory(studentListView -> {
            HistoryListViewCell cell = new HistoryListViewCell();

            cell.setOnMouseClicked(e -> {
                cell.updateSelected(false);

                if(e.getClickCount() == 2) {
                    HistoryRow selectedRow = historyListView.getSelectionModel().getSelectedItems().get(0);

                    if(selectedRow.isRunningCC && !selectedRow.endedCC.get()) {
                        return;
                    }

                    if(selectedRow.isRunningDL && !selectedRow.endedDL.get()) {
                        return;
                    }

                    if(selectedRow.isRunningCC) {
                        historyViewPresentCCResults(selectedRow);
                    } else {
                        historyViewPresentDLResults(selectedRow);
                    }
                }
            });

            return cell;
        });
    }

    protected void historyViewPresentCCResults(HistoryRow row){


        JFXDialogLayout layout = new JFXDialogLayout();
        ScrollPane scroll = new ScrollPane(new Text(row.getResultsTextCC()));
        scroll.setPadding(new Insets(8, 8,8,8));
        scroll.setFitToWidth(true);

        layout.setBody(scroll);
        layout.setMinWidth(665);

        JFXDialog dialog = new JFXDialog(root, layout, JFXDialog.DialogTransition.CENTER);

        String buttonTitle;
        EventHandler<ActionEvent> buttonHandler;

        if(row.isRunningDL) {
            buttonTitle = "See Deep Learning Results";

            buttonHandler = new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent event) {
                    dialog.close();
                    historyViewPresentDLResults(row);
                }
            };
        } else {
            buttonTitle = "Close";
            buttonHandler = new EventHandler<ActionEvent>() {

                @Override
                public void handle(ActionEvent event) {
                    dialog.close();
                }
            };
        }

        JFXButton closeButton = new JFXButton(buttonTitle.toUpperCase());
        closeButton.setTextFill(Color.WHITE);
        closeButton.setStyle("-fx-background-color: #2198F3");
        closeButton.setOnAction(buttonHandler);
        layout.setActions(closeButton);

        dialog.show();
    }

    protected void historyViewPresentDLResults(HistoryRow row){
        JFXDialogLayout layout = new JFXDialogLayout();

        ScrollPane scroll = new ScrollPane(new Text(row.getResultsTextDL()));
        scroll.setPadding(new Insets(8, 8,8,8));
        scroll.setFitToWidth(true);

        layout.setBody(scroll);
        layout.setMinWidth(665);

        JFXDialog dialogDL = new JFXDialog(root, layout, JFXDialog.DialogTransition.CENTER);

        JFXButton closeButton = new JFXButton("Close".toUpperCase());
        closeButton.setTextFill(Color.WHITE);
        closeButton.setStyle("-fx-background-color: #2198F3");
        closeButton.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                dialogDL.close();
            }
        });

        layout.setActions(closeButton);
        dialogDL.show();
    }

    @FXML protected void handleLoadBase(ActionEvent event) throws Exception {
        File f = loadArff();

        if (f != null && f.getName().endsWith(".arff")) {

            DataSource source = new DataSource(f.getAbsolutePath());
            base = new Instances(source.getDataSet());

            int[] baseIndexesOrder = new int[base.numAttributes()];

            for(int i = 0; i < base.numAttributes(); i++) {
                baseIndexesOrder[i] = i;
            }

            attOrderFilter.setAttributeIndicesArray(baseIndexesOrder);
            attOrderFilter.setInputFormat(base);

            base.setClass(base.attribute(base.numAttributes() - 1));

            updateBaseInfoSection();
        }
    }

    private Instances formatClass(Instances b) {
        if(base.classAttribute().isNumeric()) {
            return base;
        }

        Instances base = new Instances(b);

        List<String> newClassValeus = new ArrayList<>();
        for(int i = 0; i < base.classAttribute().numValues(); i++) {
            newClassValeus.add(String.valueOf(i));
        }

        String[] oldValues = new String[base.numInstances()];

        for(int i = 0; i < base.numInstances(); i++) {
            oldValues[i] = String.valueOf((int)base.instance(i).value(base.numAttributes()-1));
        }

        Attribute newClass = new Attribute(base.classAttribute().name(), newClassValeus);

        if(base.classIndex() != 0) {
            base.setClassIndex(0);
        } else {
            base.setClassIndex(1);
        }

        base.deleteAttributeAt(base.numAttributes() - 1);
        base.insertAttributeAt(newClass, base.numAttributes());
        base.setClassIndex(base.numAttributes()-1);

        HashMap map = new HashMap();
        int v = 0;

        for(int i = 0; i < base.numInstances(); i++) {
            String classVal = oldValues[i];

            if(!map.containsKey(classVal)) {
                map.put(classVal, String.valueOf(v));
                v++;
            }

            base.instance(i).setClassValue((String) map.get(classVal));
        }

        String className = base.classAttribute().name();
        base.setClassIndex(base.numAttributes() - 1);

        return base;
    }

    private File loadArff() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open ARFF file");
        FileChooser.ExtensionFilter arff = new FileChooser.ExtensionFilter("ARFF files (*.arf)",  "*.ARFF");
        fileChooser.getExtensionFilters().addAll(arff);

        return fileChooser.showOpenDialog(baseInfoClassesLabel.getScene().getWindow());
    }

    private void updateBaseInfoSection() {
        updateNameLabel();
        updateInstancesLabel();
        updateClassesLabel();
        updateAttributesLabel();
        updateClassesComboBox();
    }

    private void updateClassesComboBox() {
        ObservableList<String> options = FXCollections.observableArrayList();
        for(int i = 0; i < base.numAttributes(); i++) {
            options.add(base.attribute(i).name());
        }

        baseClassComboBox.setItems(options);
        baseClassComboBox.setDisable(false);
        baseClassComboBox.setValue(options.get(options.size() - 1));
    }

    private void updateNameLabel() {
        baseInfoNameLabel.setText("Name: " + base.relationName());
    }

    private void updateInstancesLabel() {
        StringBuilder builder = new StringBuilder("Instances: ");
        builder.append(base.numInstances());
        baseInfoInstancesLabel.setText(builder.toString());
    }

    private void updateAttributesLabel() {
        StringBuilder builder = new StringBuilder("Attributes: ");
        builder.append(base.numAttributes() - 1);
        baseInfoAttributesLabel.setText(builder.toString());
    }

    private void updateClassesLabel() {
        StringBuilder builder = new StringBuilder("Classes: ");

        if(!base.attribute(base.classIndex()).isNumeric()) {
            builder.append(base.numClasses());
        } else {
            builder.append("Not Discrete");
        }

        baseInfoClassesLabel.setText(builder.toString());
    }

    @FXML protected void handleClassComboBox(ActionEvent event) throws Exception {

        base = Filter.useFilter(base, attOrderFilter);

        for(int i = 0; i < base.numAttributes(); i++) {
            if (base.attribute(i).name() == baseClassComboBox.getValue()) {

                int[] auxAttributeOrder = new int[base.numAttributes()];

                for(int j = 0; j < base.numAttributes(); j++) {
                    auxAttributeOrder[j] = j + 1;
                }

                auxAttributeOrder[auxAttributeOrder.length - 1] = i + 1;
                auxAttributeOrder[i] = auxAttributeOrder.length;

                StringBuilder indexStr = new StringBuilder();
                for(int a = 0; a < auxAttributeOrder.length; a++) {
                    indexStr.append(auxAttributeOrder[a] + ",");
                }

                indexStr.deleteCharAt(indexStr.length() - 1);

                attOrderFilter.setAttributeIndices(indexStr.toString());
                attOrderFilter.setInputFormat(base);

                base.setClassIndex(base.numAttributes() - 1);
                Instances newBase = Filter.useFilter(base, attOrderFilter);
                base = newBase;
                base.setClassIndex(base.numAttributes()-1);

                StringBuilder builder = new StringBuilder("Classes: ");

                if(base.classAttribute().isNominal()) {
                    builder.append(base.numClasses());
                } else {
                    builder.append("Not Discrete.");
                }

                baseInfoClassesLabel.setText(builder.toString());

                return;
            }
        }
    }

    @FXML protected void handleRunButton(ActionEvent event) throws Exception {

        boolean showDialog = true;
        JFXDialogLayout layout = new JFXDialogLayout();

        if(base == null) {
            layout.setBody(new Text("Load a base file before you run." +
                    "\nYou can load a base file with the button on the top-left."));
        } else if (!checkBoxEnableDL.isSelected() && !checkBoxEnableCC.isSelected()) {
            layout.setBody(new Text("Select a classification training method to use." +
                    "\nEnable at least one with the checkbox on either\nDEEP LEARNING or CLASSIC CLASSFICATION."));
        } else {
            showDialog = false;
        }

        if (showDialog) {
            JFXDialog dialog = new JFXDialog(root, layout, JFXDialog.DialogTransition.CENTER);

            JFXButton closeButton = new JFXButton("Close");
            closeButton.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent event) {
                    dialog.close();
                }
            });

            layout.setActions(closeButton);
            dialog.show();

            return;
        }

        HistoryRow r = new HistoryRow(checkBoxEnableDL.isSelected(), checkBoxEnableCC.isSelected());

        Thread ccThread = new Thread();

        if (checkBoxEnableCC.isSelected()) {
            if (radioButtonC45.isSelected()) {
                ccThread = new Thread(() -> {
                    try {
                        runC45();
                    } catch(Exception e) {
                        e.printStackTrace();
                    }
                });

                r.classicalType = HistoryRow.CCType.C45;
            } else if (radioButtonNaiveBayes.isSelected()) {
                ccThread = new Thread(() -> {
                    try {
                        runNaiveBayes();
                    } catch(Exception e) {
                        e.printStackTrace();
                    }
                });

                r.classicalType = HistoryRow.CCType.NB;
            } else if (radioButtonNeuralNetwork.isSelected()) {
                ccThread = new Thread(() -> {
                    try {
                        runNeuralNetwork();
                    } catch(Exception e) {
                        e.printStackTrace();
                    }
                });

                r.classicalType = HistoryRow.CCType.NN;
            } else if (radioButtonRuleBased.isSelected()) {
                ccThread = new Thread(() -> {
                    try {
                        runRuleBased();
                    } catch(Exception e) {
                        e.printStackTrace();
                    }
                });

                r.classicalType = HistoryRow.CCType.RB;
            }
        }

        Thread dlThread = new Thread();


        if (checkBoxEnableDL.isSelected()) {
            dlThread = new Thread(() -> {
                try {
                    runDeepLearning();
                } catch(Exception e) {
                    e.printStackTrace();
                }
            });
        }

        historyList.add(r);
        dlThread.start();
        ccThread.start();
    }

    @FXML protected void updateClassifierOptions(ActionEvent event) throws Exception {
        parametersBox.getChildren().clear();

        if(radioButtonC45.isSelected()) {
            addOption("Confidence Factor", sliderConfidenceFactorC45);
            addOption("Min. intances per leaf", sliderMinLeaftC45);
        } else if(radioButtonNaiveBayes.isSelected()) {
            addOption("No Parameters", null);
        } else if(radioButtonNeuralNetwork.isSelected()) {
            addOption("Learning Rate", sliderLearningrRateNN);
            addOption("Momentum", sliderMomentumNN);
            addOption("Validation Set Size", slideValidationSetSizeNN);
            addOption("Hidden Layers", textFieldHiddenLayersNN);
        } else if(radioButtonRuleBased.isSelected()) {
            addOption("Min. Bucket Size", sliderMinBucketSizeRB);
        }
    }

    protected void addOption(String title, Node n) {
        Label titleLabel = new Label(title);

        titleLabel.setFont(new Font("Roboto", 15));
        titleLabel.setTextFill(Color.WHITE);

        parametersBox.getChildren().add(titleLabel);
        if(n != null) parametersBox.getChildren().add(n);
    }

    protected void runC45() throws Exception{
        int historyIndex = historyList.size() - 1;

        try {
            J48 c45 = new J48();
            c45.setConfidenceFactor(((float)sliderConfidenceFactorC45.getValue())/100);
            c45.setMinNumObj((int)sliderMinLeaftC45.getValue());

            c45.buildClassifier(base);
            Evaluation e = new Evaluation(base);
            e.crossValidateModel(c45, base, 10, new Random(1));

            Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e, c45));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }

    protected void runNaiveBayes() throws Exception {
        int historyIndex = historyList.size() - 1;
        try {
            NaiveBayes NB = new NaiveBayes();

            NB.buildClassifier(base);
            Evaluation e = new Evaluation(base);
            e.crossValidateModel(NB, base, 10, new Random(1));

            Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e, NB));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }

    protected void runRuleBased() throws Exception {
        int historyIndex = historyList.size() - 1;

        try {
            OneR RB = new OneR();
            RB.setMinBucketSize((int)sliderMinBucketSizeRB.getValue());

            RB.buildClassifier(base);

            Evaluation e = new Evaluation(base);
            e.crossValidateModel(RB, base, 10, new Random(1));

            Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e, RB));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }

    protected void runNeuralNetwork() throws Exception {
        int historyIndex = historyList.size() - 1;
        try {
            MultilayerPerceptron NN = new MultilayerPerceptron();
            NN.setLearningRate(sliderLearningrRateNN.getValue()/100);
            NN.setMomentum(sliderMomentumNN.getValue()/100);
            NN.setValidationSetSize((int)slideValidationSetSizeNN.getValue());
            NN.setHiddenLayers(textFieldHiddenLayersNN.getText());

            NN.buildClassifier(base);
            Evaluation e = new Evaluation(base);
            e.crossValidateModel(NN, base, 10, new Random(1));

            Platform.runLater(() -> finishCCTaskOfIndex(historyIndex, e, NN));
        } catch (Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }

    protected void runDeepLearning() throws Exception {
        int historyIndex = historyList.size() - 1;
        int nAttr = base.numAttributes() - 1;

        try {
            NeuralNetConfiguration.ListBuilder DLBuilder = new NeuralNetConfiguration.Builder()
                .learningRate(sliderLearningRateDL.getValue()/100)
                .iterations(1)
                .seed(1)
                .regularization(true)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .activation(new ActivationReLU())
                    .list()
                    .backprop(true)
                    .pretrain(false);

            String[] hiddenLayersParams = textFieldHiddenLayersDL.getText().split(" ");
            int[] convLayersSize = new int[hiddenLayersParams.length];

            for(int i = 0; i < hiddenLayersParams.length; i++) {
                if(hiddenLayersParams[i].equals("a")) {
                    convLayersSize[i] = (nAttr + base.numClasses())/2;
                } else if(hiddenLayersParams[i].equals("i")) {
                    convLayersSize[i] = nAttr;
                } else if(hiddenLayersParams[i].equals("o")) {
                    convLayersSize[i] = base.numClasses();
                } else {
                    convLayersSize[i] = Integer.parseInt(hiddenLayersParams[i]);
                }
            }

            List<Layer> layersList = new ArrayList<>();

            for(int i = 0; i < convLayersSize.length; i++) {
                Convolution1DLayer convolutionalLayer = new Convolution1DLayer();
                convolutionalLayer.setLearningRate(sliderLearningRateDL.getValue()/100);
                convolutionalLayer.setKernelSize(new int[]{(int)sliderKernelXDL.getValue() ,(int)sliderKernelYDL.getValue()});
                convolutionalLayer.setStride(new int[]{(int)sliderStrideXDL.getValue() ,(int)sliderStrideYDL.getValue()});
                convolutionalLayer.setLayerName("CL" + i);
                convolutionalLayer.setActivationFn(new ActivationReLU());
                convolutionalLayer.setNOut(convLayersSize[i]);
                convolutionalLayer.setWeightInit(WeightInit.RELU);
                convolutionalLayer.setConvolutionMode(ConvolutionMode.Truncate);
                convolutionalLayer.setPadding(new int[]{(int)sliderKernelXDL.getValue() * 2, (int)sliderKernelYDL.getValue() * 2});

                if(i == 0) {
                    convolutionalLayer.setNIn(nAttr);
                } else {
                    convolutionalLayer.setNIn(convLayersSize[i - 1]);
                }

                Subsampling1DLayer poolingLayer = new Subsampling1DLayer.Builder()
                        .name("PL" + 1)
                        .kernelSize(2)
                        .stride(2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build();

                layersList.add(convolutionalLayer);
                layersList.add(poolingLayer);
            }

            OutputLayer outputLayer = new OutputLayer.Builder()
                    .name("Output Layer")
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .nOut(base.numClasses())
                    .build();

            layersList.add(outputLayer);

            for(int i = 0; i < layersList.size(); i++) {
                DLBuilder = DLBuilder.layer(i, layersList.get(i));
            }

            CSVSaver csvSaver = new CSVSaver();
            csvSaver.setInstances(this.formatClass(base));
            csvSaver.setFile(new File("temp.csv"));
            csvSaver.setNoHeaderRow(true);
            csvSaver.writeBatch();

            RecordReader reader = new CSVRecordReader();
            reader.initialize(new FileSplit(new File("temp.csv")));

            DataSetIterator iterator =  new RecordReaderDataSetIterator(reader, base.numInstances(), base.classIndex(), base.numClasses());
            DataSet data = iterator.next();

            data.shuffle();
            SplitTestAndTrain split = data.splitTestAndTrain(0.7);

            DataSet trainData = split.getTrain();
            DataSet testData = split.getTrain();

            DataNormalization normalization = new NormalizerStandardize();

            normalization.fit(trainData);
            normalization.transform(trainData);
            normalization.transform(testData);

            MultiLayerConfiguration.Builder builder = DLBuilder.setInputType(InputType.convolutionalFlat(trainData.numExamples(), nAttr, 1));
            builder.pretrain(false);
            builder.backprop(true);

            System.out.println(base.numAttributes() * base.numInstances());

            MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
            model.init();
            model.fit(trainData);

            org.deeplearning4j.eval.Evaluation e = new org.deeplearning4j.eval.Evaluation(base.numClasses());
            INDArray output = model.output(testData.getFeatureMatrix());
            e.eval(split.getTest().getLabels(), output);

            Platform.runLater(() -> finishDLTaskOfIndex(historyIndex, e, model));
        } catch(Exception e) {
            e.printStackTrace();
            Platform.runLater(() -> finishTaskOfIndexWithError(historyIndex));
        }
    }

    protected void finishDLTaskOfIndex(int index, org.deeplearning4j.eval.Evaluation eval, MultiLayerNetwork net) {
        historyList.get(index).endedDL.set(true);
        historyList.get(index).evalDL = eval;
        historyList.get(index).classifierDL = net;
    }

    protected void finishCCTaskOfIndex(int index, Evaluation eval, Classifier c) {
        historyList.get(index).endedCC.set(true);
        historyList.get(index).evalCC = eval;
        historyList.get(index).classifierCC = c;
    }

    protected void finishTaskOfIndexWithError(int index) {
        historyList.get(index).fail.set(true);
    }
}