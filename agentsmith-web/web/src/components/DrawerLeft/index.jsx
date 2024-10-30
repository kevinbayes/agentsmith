import React from 'react';

import BaseComponent from "../BaseComponent/index.jsx";
import {Bloc} from "./bloc.js";
import {withAuth0} from "@auth0/auth0-react";
import {
    Box, IconButton, Stack,
} from "@mui/material";
import {withAppContext} from "../../pages/context.jsx";
import {H1Header, H1HeaderDrawer} from "../Typography/index.jsx";
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';



class DrawerLeft extends BaseComponent {

    constructor(props) {
        super(props);
        this.setBloc(new Bloc({
            leftDrawer: 350,
        }, props.globalContext));

        this.state = {};
    }

    componentDidMount() {
        super.componentDidMount();
    }

    render() {

        const { initialised } = this.state;

        return (
            <Box sx={{padding: "60pt 7pt 7pt 7pt",}}>
                <Stack spacing={2}>
                    <Box sx={{ display: "flex", }}>Left Drawer</Box>
                </Stack>
            </Box>
        );
    }
}

export default withAuth0(withAppContext(DrawerLeft));