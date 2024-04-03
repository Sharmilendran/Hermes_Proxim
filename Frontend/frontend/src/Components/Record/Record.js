import { useState } from "react";
import { Button } from "@mui/material";
import ClassificationList from "../ClassificationList/ClassificationList";
import ProximityList from "../ProximityList/ProximityList";

const Record = () => {
    const [classifications, setClassifications] = useState([]);

    const handleGetClassificationData = () => {
        fetch('http://localhost:5000/predict')
            .then((res) => {
                return res.json();
            })
            .then((data) => {
                console.log(data);
                setClassifications(data);
            });
    }
    return (
        <>
            <Button onClick={handleGetClassificationData}> Record button</Button>
            <ClassificationList classifications={classifications}/>
            <ProximityList proximities={classifications}/>

        </>
    );
};

export default Record;