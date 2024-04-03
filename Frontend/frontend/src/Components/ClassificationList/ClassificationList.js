import { TableHead } from "@mui/material";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableRow from "@mui/material/TableRow";
import "./ClassificationList.css"

const ClassificationList = ({ classifications }) => {

    return (
        <div className="teamMemberContainer">
            <div>
                <TableContainer>
                    <Table size="medium" aria-label="a dense table" border="solid">
                        <TableHead>
                            <TableCell>
                                Second
                            </TableCell>
                            <TableCell>
                                Classification
                            </TableCell>
                        </TableHead>
                        <TableBody>
                            {classifications?.classifications?.map((data) => (
                                <TableRow
                                    key={data.id}
                                    sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
                                >
                                    <TableCell>
                                        <div className="event">
                                            <label className="member-name">{data[0]}</label>
                                        </div>
                                    </TableCell>
                                    <TableCell>
                                        <div className="event">
                                            <label className="member-name">{data[1]}</label>
                                        </div>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            </div>
        </div>
    );
};

export default ClassificationList;