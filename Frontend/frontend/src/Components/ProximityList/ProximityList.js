import { TableHead } from "@mui/material";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableRow from "@mui/material/TableRow";

const ProximityList = ({ proximities }) => {
return (
    <div className="teamMemberContainer">
        <div>
            <TableContainer style={{ height: 500, width: 500 }}>
                <Table size="medium" aria-label="a dense table" border="solid">
                    <TableHead>
                        <TableCell>
                            Second
                        </TableCell>
                        <TableCell>
                            Proximity
                        </TableCell>
                    </TableHead>
                    <TableBody>
                        {proximities?.proximity?.map((data) => (
                            <TableRow
                                key={data.id}
                                sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
                            >
                                <TableCell>
                                    <div className="event">
                                        <label className="member-name">{data.Classifications}</label>
                                    </div>
                                </TableCell>
                                <TableCell>
                                    <div className="event">
                                        <label className="member-name">{data.Proximity}</label>
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

export default ProximityList;