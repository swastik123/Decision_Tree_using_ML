import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;

public class ID3 {
	
	private int[][] trainData, testData, validateData;
	private Map<Integer, String> dTree = new HashMap<Integer, String>();
	private Map<Integer, String> hRow = new HashMap<Integer, String>();
	private Map<String, Integer> headRowReverse = new HashMap<String, Integer>();
	private BufferedWriter logWriter;
	


	public Map<Integer, String> gethRow() {
		return this.hRow;
	}

	public void sethRow(Map<Integer, String> hRow) {
		this.hRow = hRow;
	}

	public Map<String, Integer> getheadRowReverse() {
		return this.headRowReverse;
	}

	public void setheadRowReverse(
			Map<String, Integer> headRowReverse) {
		this.headRowReverse = headRowReverse;
	}

	public int[][] getTrainData() {
		return this.trainData;
	}

	public void setTrainData(int[][] trainData) {
		this.trainData = trainData;
	}

	public int[][] getTestData() {
		return this.testData;
	}

	public void setTestData(int[][] testData) {
		this.testData = testData;
	}

	public int[][] getValidationData() {
		return this.validateData;
	}

	public void setValidationData(int[][] validationData) {
		this.validateData = validationData;
	}

	public Map<Integer, String> getDTree() {
		return this.dTree;
	}

	public void setDTree(Map<Integer, String> decisionTree) {
		this.dTree = decisionTree;
	}

	public ID3() throws IOException {
		logWriter = new BufferedWriter(new FileWriter("ID3OP.txt"));
	}

	public ID3(String random) throws IOException {
		logWriter = new BufferedWriter(new FileWriter("ID3RandomOP.txt"));
	}


	private int[][] readInput(String fileName) throws IOException {
		int[][] inputData;
		int length = 0, width = 0;
		String record = null;
		System.out.println();
		
		BufferedReader br = new BufferedReader(new FileReader(fileName));

		String header = br.readLine();
		while (br.readLine() != null)
			length++;
		br.close();
		StringTokenizer st = new StringTokenizer(header, ",");
		int i = 0;
		String str = null;
		while (st.hasMoreTokens()) {
			str = st.nextToken();
			this.hRow.put(i, str);
			this.headRowReverse.put(str, i++);
		}
		width = this.hRow.size();
		inputData = new int[length][width];
		
		br = new BufferedReader(new FileReader(fileName));
		br.readLine();
		length = 0;
		while ((record = br.readLine()) != null) {
			width = 0;
			st = new StringTokenizer(record, ",");
			while (st.hasMoreTokens()) {
				inputData[length][width++] = Integer.parseInt(st.nextToken());
			}
		length++;
		}
		return inputData;
	}

	public void inputDataToTree(String trainingDataPath, String validationDataPath, String testDataPath) throws IOException {
		setTrainData(readInput(trainingDataPath));
		setValidationData(readInput(validationDataPath));
		setTestData(readInput(testDataPath));
	}


	private double getEntropy(int[][] datas) throws Exception {

		if (datas.length == 0)
			return 0;

		int positive = 0, negative = 0;
		for (int i = 0; i < datas.length; i++) {
			if (datas[i][datas[i].length - 1] == 0) {
				negative++;
			} else {
				positive++;
			}
		}

		double entropy = (negative == 0 ? 0 : -(((double) negative / datas.length)* Math.log10((double) negative / datas.length) / Math.log10(2))) + (positive == 0 ? 0 : -(((double) positive / datas.length) * Math.log10((double) positive / datas.length) / Math.log10(2)));
		return entropy;
	}


	private double getEntropy(int negativeSamplesCnt, int positiveSamplesCnt, int totalSamplesCnt) throws Exception {

		if (totalSamplesCnt == 0)
			return 0;

		double entropy = (negativeSamplesCnt == 0 ? 0 : -(((double) negativeSamplesCnt / totalSamplesCnt) * Math.log10((double) negativeSamplesCnt / totalSamplesCnt) / Math.log10(2))) + (positiveSamplesCnt == 0 ? 0 : -(((double) positiveSamplesCnt / totalSamplesCnt)* Math.log10((double) positiveSamplesCnt / totalSamplesCnt) / Math.log10(2)));
		
		return entropy;
	}

	private void createRandomDecisionTree(int[][] data, Map<Integer, String> header, int position) throws Exception {

		if (position == 1) 
			setDTree(new HashMap<Integer, String>());
		double rootEntropy = getEntropy(data);
			if (rootEntropy == 0) {
					getDTree().put(position, "" + data[0][data[0].length - 1]);
			} else if (header.size() == 1) {
				int neg = 0, pos = 0;
				for (int i = 0; i < data.length; i++) {
					if (data[i][data[i].length - 1] == 0) {
						neg++;
					} else {
							pos++;
							}
				}
	
				if (neg > pos) {
					getDTree().put(position, "0");
				} else {
					getDTree().put(position, "1");
				}
		} else {
				double maxGain = -2;
				Random random = new Random();
				int max = data[0].length - 1, min = 0;
				int GainAttribute = random.nextInt(max - min + 1) + min;
				int GainLeftNodeCnt = -1;
				int GainRightNodeCnt = -1;
				int GainMajorityClass = -1;

				while (!header.containsKey(GainAttribute) || checkNode(position, header.get(GainAttribute)))
					GainAttribute = random.nextInt(max - min + 1) + min;
				
				int neg = 0, pos = 0, negNeg = 0, negPos = 0, posNeg = 0, posPos = 0;
				for (int j = 0; j < data.length; j++) {
						if (data[j][GainAttribute] == 0) {
								neg++;
								if (data[j][data[j].length - 1] == 0) {
									negNeg++;
								} else {
									negPos++;
								}
						} else {
							pos++;
							if (data[j][data[j].length - 1] == 0) {
								posNeg++;
							} else {
								posPos++;
							}
						}
				}
			double nEntropy = 0, pEntropy = 0, entropy = 0, gain = 0;

			nEntropy = getEntropy(negNeg, negPos, neg);
			pEntropy = getEntropy(posNeg, posPos, pos);
			entropy = ((double) neg / (neg + pos)) * nEntropy + ((double) pos / (neg + pos)) * pEntropy;
			gain = entropy;
	
			maxGain = gain;
			GainLeftNodeCnt = neg;
			GainRightNodeCnt = pos;
			GainMajorityClass = ((negNeg + posNeg) > (negPos + posPos)) ? 0 : 1;

			getDTree().put(position, header.get(GainAttribute));
			int[][] leftData = new int[GainLeftNodeCnt][data[0].length];
			int[][] rightData = new int[GainRightNodeCnt][data[0].length];
			Map<Integer, String> leftHeader = (Map<Integer, String>) ((HashMap<Integer, String>) header).clone();
			leftHeader.remove(GainAttribute);
			Map<Integer, String> rightHeader = (Map<Integer, String>) ((HashMap<Integer, String>) header).clone();
			rightHeader.remove(GainAttribute);
			int leftCounter= 0, rightCounter = 0;
			for (int i = 0; i < data.length; i++) {
					if (data[i][GainAttribute] == 0) {
						for (int j = 0; j < data[i].length; j++) {
							leftData[leftCounter][j] = data[i][j];
						}
						leftCounter++;
					} else {
						for (int j = 0; j < data[0].length; j++) {
							rightData[rightCounter][j] = data[i][j];
						}
						rightCounter++;
					}	
			}
			if (GainLeftNodeCnt == 0) {
				getDTree().put(position * 2, "" + GainMajorityClass);
			} else {
				createRandomDecisionTree(leftData, leftHeader, position * 2);
			}
			if (GainRightNodeCnt == 0) {
				getDTree().put(position * 2 + 1, "" + GainMajorityClass);
			} else {
				createRandomDecisionTree(rightData, rightHeader, position * 2 + 1);
			}
		}

}
	

	private boolean checkNode(int Position, String string) {
		Map<Integer, String> a = getDTree();
		while (Position >= 2) {
			if (Position % 2 == 0) {
				if (getDTree().get(Position / 2).equals(string))
					return true;
			} else {
				if (getDTree().get((Position - 1) / 2).equals(string))
					return true;
			}

			Position = Position / 2;
		}
		if (a.size() > 0)
			if (getDTree().get(1).equals(string))
				return true;
		return false;
	}

	public void createDecisionTree(int[][] data, Map<Integer, String> header, int Position) throws Exception {

		if (Position == 1) {
			setDTree(new HashMap<Integer, String>());
		}

		double firstEntropy = getEntropy(data);

		if (firstEntropy == 0) {
			getDTree().put(Position, "" + data[0][data[0].length - 1]);
		} else if (header.size() == 1) {
			int neg = 0, pos = 0;
			for (int i = 0; i < data.length; i++) {
				if (data[i][data[i].length - 1] == 0) {
					neg++;
				} else {
					pos++;
				}
			}
			if (neg > pos) {
				getDTree().put(Position, "0");
			} else {
				getDTree().put(Position, "1");
			}
		} else {
			double Gain = -2;
			int GainAttribute = -1;
			int GainLeftNodeCnt = -1;
			int GainRightNodeCnt = -1;
			int GainMajorityClass = -1;

			for (int i = 0; i < data[0].length - 1; i++) {
				if (!header.containsKey(i))
					continue;
				int neg = 0, pos = 0, negNeg = 0, negPos = 0, posNeg = 0, posPos = 0;
				for (int j = 0; j < data.length; j++) {
					if (data[j][i] == 0) {
						neg++;
						if (data[j][data[j].length - 1] == 0) {
							negNeg++;
						} else {
							negPos++;
						}
					} else {
						pos++;
						if (data[j][data[j].length - 1] == 0) {
							posNeg++;
						} else {
							posPos++;
						}
					}
				}
				double nEntropy = 0, pEntropy = 0, entropy = 0, gain = 0;

				nEntropy = getEntropy(negNeg, negPos, neg);
				pEntropy = getEntropy(posNeg, posPos, pos);
				entropy = ((double) neg / (neg + pos)) * nEntropy + ((double) pos / (neg + pos)) * pEntropy;
				gain = firstEntropy - entropy;
				if (gain > Gain) {
					Gain = gain;
					GainAttribute = i;
					GainLeftNodeCnt = neg;
					GainRightNodeCnt = pos;
					GainMajorityClass = ((negNeg + posNeg) > (negPos + posPos)) ? 0 : 1;
				}
			}
			getDTree().put(Position, header.get(GainAttribute));
			int[][] leftData = new int[GainLeftNodeCnt][data[0].length], rightData = new int[GainRightNodeCnt][data[0].length];
			Map<Integer, String> leftHeader = (Map<Integer, String>) ((HashMap<Integer, String>) header).clone();
			leftHeader.remove(GainAttribute);
			Map<Integer, String> rightHeader = (Map<Integer, String>) ((HashMap<Integer, String>) header).clone();
			rightHeader.remove(GainAttribute);
			int lCounter = 0, rCounter = 0;
			for (int i = 0; i < data.length; i++) {
				if (data[i][GainAttribute] == 0) {
					for (int j = 0; j < data[i].length; j++) {
						leftData[lCounter][j] = data[i][j];
					}
					lCounter++;
				} else {
					for (int j = 0; j < data[0].length; j++) {
						rightData[rCounter][j] = data[i][j];
					}
					rCounter++;
				}
			}
			if (GainLeftNodeCnt == 0) {
				getDTree().put(Position * 2, "" + GainMajorityClass);
			} else {
				createDecisionTree(leftData, leftHeader, Position * 2);
			}
			if (GainRightNodeCnt == 0) {
				getDTree().put(Position * 2 + 1,
						"" + GainMajorityClass);
			} else {
				createDecisionTree(rightData, rightHeader, Position * 2 + 1);
			}
		}
	}

	public double testDecisionTree(Map<Integer, String> decisionTree, int[][] testData) throws IOException {
		int correct = 0, wrong = 0;
		boolean isCorrect = false;
		for (int i = 0; i < testData.length; i++) {
			isCorrect = estimation(decisionTree, testData[i]);
			if (isCorrect)
				correct++;
			else
				wrong++;
		}

		return ((double) correct / (correct + wrong)) * 100;
	}


	private boolean estimation(Map<Integer, String> decisionTree, int[] testSample) throws IOException {
		int nodePos = 1;
		while (!decisionTree.get(nodePos).equals("0") && !decisionTree.get(nodePos).equals("1")) {
			nodePos = testSample[this.headRowReverse.get(decisionTree.get(nodePos))] == 0 ? nodePos * 2 : nodePos * 2 + 1;
		}
		if (decisionTree.get(nodePos).equals("" + testSample[testSample.length - 1]))
			return true;
		else {
			
			return false;
		}
	}
	private void logDecisionTree(int nodePos, String indentation,String condition) throws IOException {
		if (getDTree().containsKey(nodePos)) {
			if (!getDTree().containsKey(2 * nodePos) && !getDTree().containsKey(2 * nodePos + 1)) {
				System.out.println(indentation.substring(1) + condition + getDTree().get(nodePos));
				logWriter.write(indentation.substring(1) + condition + getDTree().get(nodePos));
				logWriter.newLine();
			} else {
				System.out.println((indentation.length() > 0 ? indentation.substring(1) : indentation) + condition);
				logWriter.write((indentation.length() > 0 ? indentation.substring(1) : indentation) + condition);
				logWriter.newLine();
				logDecisionTree(nodePos * 2, indentation + "| ", getDTree().get(nodePos) + " = 0 : ");
				logDecisionTree(nodePos * 2 + 1, indentation + "| ",getDTree().get(nodePos) + " = 1 : ");
			}
		}
	}

	
	private void log(String str) throws IOException {
		System.out.println(str);
		logWriter.write(str);
		logWriter.newLine();
	}
	
	public static void main(String[] args) throws Exception {
		
		try {
			PrintStream out = new PrintStream(new FileOutputStream("ID3OP.txt"));
			System.setOut(out);
		}

		catch (IOException e1) {
			System.out.println("Invalid input received");
		}
		
		int prunefact = 0; 
		String trainingDataPath = "", validationDataPath = "", testDataPath = "";//
		boolean print = false;

		
	
		prunefact = Integer.parseInt("50");
		trainingDataPath = "D:\\train.csv";
		validationDataPath ="D:\\validation.csv";
		testDataPath ="D:\\test.csv";
		
		

		ID3 tree = new ID3();
		
		ID3 randomtree = new ID3("random");
		
		try {
			tree.inputDataToTree(trainingDataPath, validationDataPath,testDataPath);

			tree.createDecisionTree(tree.getTrainData(), tree.gethRow(), 1); 
			
				
		

			tree.log("");
			tree.log("Before Pruning Tree: ");
			tree.logDecisionTree(1, "", "");
			double beforePruningAccuracy = tree.testDecisionTree(tree.getDTree(), tree.getTestData()); 
				
				            tree.log("");
							tree.log("After Pruning Tree: ");
							tree.logDecisionTree(1, "", "");
				double afterPruningAccuracyRand = tree.testDecisionTree(tree.getDTree(), tree.getTestData());	
			
		} catch (Exception e) {
			throw e;
		}
		
		try {
			PrintStream out1 = new PrintStream(new FileOutputStream("ID3RandomOP.txt"));
			System.setOut(out1);
			
		}

		catch (IOException e1) {
			System.out.println("Invalid input recieved");
		}
		
		try{

			randomtree.inputDataToTree(trainingDataPath, validationDataPath,testDataPath);	
			randomtree.createRandomDecisionTree(randomtree.getTrainData(), randomtree.gethRow(), 1);
				
			randomtree.log("");
			randomtree.log("Before Pruning Tree: ");
			randomtree.logDecisionTree(1, "", "");
				double beforePruningAccuracyRand = randomtree.testDecisionTree(randomtree.getDTree(), randomtree.getTestData()); 
							randomtree.log("");
				randomtree.log("After Pruning Tree: ");
				randomtree.logDecisionTree(1, "", "");
				double afterPruningAccuracyRand = randomtree.testDecisionTree(randomtree.getDTree(), randomtree.getTestData());	
				
			
		}catch (Exception e) {
			throw e;
		}

	}
	
}