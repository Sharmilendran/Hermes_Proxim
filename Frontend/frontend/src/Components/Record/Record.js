import { useState } from "react";
import { Button } from "@mui/material";
import ClassificationList from "../ClassificationList/ClassificationList";
import ProximityList from "../ProximityList/ProximityList";
import "bootstrap/dist/css/bootstrap.min.css";


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
            })
            .catch((e) => {
                console.log(e)
            });
    }
    return (
        <div className="container mx-5 my-5">
            <Button onClick={handleGetClassificationData}> Record button</Button>

                <div className="row my-5">
                    <div className="col">
                        <ClassificationList classifications={classifications} />
                    </div>
                    <div className="col">
                        <ProximityList proximities={classifications} />
                    </div>
                </div>
            </div>
    );
};

export default Record;