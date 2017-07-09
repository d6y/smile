package examples.tree

import smile.classification.DecisionTree
import smile.data.{Attribute, NumericAttribute, NominalAttribute, AttributeDataset}
import smile.data.parser.{ArffParser, DelimitedTextParser}
import smile.validation.LOOCV
import smile.math.Math
import java.io.File

object TreePrinter {

  type Node = DecisionTree#Node

  def print(data: AttributeDataset, node: Node): String = {
      lazy val predictedClassLabel = data.response().asInstanceOf[NominalAttribute].values()(node.output)

      // Non-leaf calculations
      lazy val featureNamme = data.colnames(node.splitFeature)
      lazy val attributeToSplitOn = data.attributes()(node.splitFeature)
      lazy val (comparisonOperator, comparisonValue) = attributeToSplitOn.getType() match {
        case Attribute.Type.NOMINAL => "="  -> attributeToSplitOn.asInstanceOf[NominalAttribute].values()(node.splitValue.toInt)
        case Attribute.Type.NUMERIC => "<=" -> node.splitValue.toString
        case at => sys.error(s"Unrecognised AttributeType: $at") // String and Date are defined in ARFF, but I'm not yet using them
      }

      (node.trueChild, node.falseChild) match {
        case (null, null) => predictedClassLabel // leaf node
        case (t,    null) => s"IF ${featureNamme} ${comparisonOperator} ${comparisonValue} THEN ${print(data, t)} ENDIF"
        case (null, f   ) => s"IF NOT(${featureNamme} ${comparisonOperator} ${comparisonValue}) THEN ${print(data, f)} ENDIF"
        case (t,    f   ) => s"IF ${featureNamme} ${comparisonOperator} ${comparisonValue} THEN ${print(data, t)} ELSE ${print(data,f)} ENDIF"
      }
  }
}

object TreeExample {

  def weather(): Unit = {
      val file = smile.data.parser.IOUtils.getTestDataFile("weka/weather.nominal.arff")
      val responseIndex = 4
      
      val arffParser = new ArffParser()
      arffParser.setResponseIndex(responseIndex)
      val weather: AttributeDataset = arffParser.parse(file)

      val (x, y) = weather.unzipInt

      import TreePrinter._

      val n = x.length
      val loocv = new LOOCV(n)

      val errors = for (i <- 0 until n) yield {
        val trainx = Math.slice(x, loocv.train(i))
        val trainy = Math.slice(y, loocv.train(i))
        
        val tree = new DecisionTree(weather.attributes(), trainx, trainy, 3)

        println( print(weather, tree.getRoot()) )

        if (y(loocv.test(i)) != tree.predict(x(loocv.test(i)))) 1 else 0
      }
          
      System.out.println("Decision Tree error = " + errors.sum);
      //assertEquals(5, error);
  }

  def titanic(trainFile: File, testFile: File, outFile: File): Unit = {
      val responseIndex = 1

      val arffParser = new ArffParser()
      arffParser.setResponseIndex(responseIndex)
      val passengers: AttributeDataset = arffParser.parse(trainFile)

      val (x, y) = passengers.unzipInt


      val tree = new DecisionTree(passengers.attributes(), x, y, 3)

      import TreePrinter._
      val rules =  print(passengers, tree.getRoot())

      val correctCount = (x zip y).map { case (passenger, outcome) => tree.predict(passenger) == outcome }.filter(identity).length

      println(
        s"""
        Total: ${x.length}
        Correct: ${correctCount}
        Percent: ${100.0 * (correctCount.toDouble / x.length.toDouble)}

        Rules:
        $rules
        """
      )

      // Write to Kaggle output format:
      val testData: AttributeDataset = new ArffParser().parse(testFile)
      val test = testData.unzip

      import purecsv.unsafe._
      case class Prediction(id: Int, survived: Int)
      test.map{ example => Prediction(example(0).toInt, tree.predict(example)) }.toSeq.writeCSVToFile(outFile, header=Some(List("PassengerId", "Survived")))
  }

  def main(args: Array[String]): Unit = {
   titanic(
      new File("/Users/richard/Developer/titanic/modified/train-mod.arff"),
      new File("/Users/richard/Developer/titanic/modified/test-mod.arff"),
      new File("/Users/richard/Developer/titanic/modified/predict.csv")
    )
  }
}
