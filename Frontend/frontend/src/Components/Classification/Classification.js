import "./Classification.css";
import { useState, useEffect } from "react";
import ClassificationList from "../ClassificationList/ClassificationList";

const GetClassification = () => {
    const [classifications, setClassifications] = useState();

      useEffect(() => {
      fetch('http://localhost:5000/predict')
        .then(response => response.json())
        .then(json => setClassifications(json))
        .catch(error => console.error(error));
    }, []);

  // const testFunc = () => {
  //     // setTest2("test")
  //     console.log("Hello Classify")
  // }
  return (
    <div >
      <ClassificationList messages={classifications} />
    </div>
  );
};

export default GetClassification;