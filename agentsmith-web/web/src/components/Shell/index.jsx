import React from 'react';

import MuiAppBar from '@mui/material/AppBar';

import BaseComponent from "../BaseComponent/index.jsx";
import {Bloc} from "./bloc.js";
import {useAuth0, withAuth0} from "@auth0/auth0-react";
import {
    Box,
    Container, CssBaseline, Drawer,
    IconButton,
    Menu, MenuItem,
    Toolbar,
    Typography
} from "@mui/material";
import {AccountCircle} from "@mui/icons-material";
import MenuIcon from '@mui/icons-material/Menu';
import {withAppContext} from "../../pages/context.jsx";
import DrawerRight from "../DrawerRight/index.jsx";
import DrawerLeft from "../DrawerLeft/index.jsx";
import {FormattedMessage} from "react-intl";

const LogoutMenuItem = () => {
    const { logout } = useAuth0();

    return (
        <MenuItem onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}>Logout</MenuItem>
    );
};



class Shell extends BaseComponent {

    constructor(props) {
        super(props);
        this.setBloc(new Bloc({
            leftDrawer: 0,
            rightDrawer: 50,
        }, props.globalContext));

        this.state = {};
    }

    componentDidMount() {
        super.componentDidMount();
    }

    __logout = (event) => {
        this.setState({anchorEl: event.currentTarget});
    };

    __toggleDrawer = (event) => {
        const { leftDrawer, } = this.state;
        this.setState({leftDrawer: leftDrawer > 0 ? 0 : 260});
    };

    __showProfile = (event) => {
        this.setState({anchorEl: event.currentTarget});
    };

    __openAccount = (event) => {
        this.setState({anchorEl: event.currentTarget});
    };

     __handleMenu = (event) => {
        this.setState({anchorEl: event.currentTarget});
    };

    __handleMenuClose = () => {
        this.setState({anchorEl: null});
    };

    render() {

        const { anchorEl, leftDrawer, rightDrawer, } = this.state;

        return (
            <>
                <MuiAppBar position="sticky" elevation={0} sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
                    <Container maxWidth={false}>
                        <Toolbar  disableGutters>
                            <Box>
                                <IconButton
                                    size="large"
                                    aria-label="toggle drawer"
                                    aria-controls="toggle-drawer"
                                    aria-haspopup="true"
                                    onClick={this.__toggleDrawer}
                                    color="inherit"
                                >
                                    <MenuIcon/>
                                </IconButton>
                            </Box>
                            <Typography variant="h6" component="div"
                                        sx={{flexGrow: 1}}>
                                <FormattedMessage id={'title'} defaultMessage={"AgentSmith"} />
                            </Typography>
                            <div>
                                <IconButton
                                    size="large"
                                    aria-label="account of current user"
                                    aria-controls="menu-appbar"
                                    aria-haspopup="true"
                                    onClick={this.__handleMenu}
                                    color="inherit"
                                >
                                    <AccountCircle/>
                                </IconButton>
                                <Menu
                                    id="menu-appbar"
                                    anchorEl={anchorEl}
                                    anchorOrigin={{
                                        vertical: 'top',
                                        horizontal: 'right',
                                    }}
                                    keepMounted
                                    transformOrigin={{
                                        vertical: 'top',
                                        horizontal: 'right',
                                    }}
                                    open={Boolean(anchorEl)}
                                    onClose={this.__handleMenuClose}
                                >
                                    <MenuItem
                                        onClick={this.__showProfile}>
                                        <FormattedMessage id={'profile'} defaultMessage={"Profile"} /></MenuItem>
                                    <MenuItem onClick={this.__openAccount}>
                                        <FormattedMessage id={'my.account'} defaultMessage={"Account"} /></MenuItem>
                                    <LogoutMenuItem />
                                </Menu>
                            </div>
                        </Toolbar>
                    </Container>
                </MuiAppBar>
                <Box sx={{ padding: "24px 12px", }}>
                    {this.props.children}
                </Box>
                <Drawer
                    sx={{
                        width: leftDrawer,
                        flexShrink: 0,
                        '& .MuiDrawer-paper': {
                            width: leftDrawer,
                            boxSizing: 'border-box',
                        },
                    }}
                    variant="persistent"
                    anchor="left"
                    open={true}
                >
                  <DrawerLeft size={leftDrawer} />
                </Drawer>
                <Drawer
                    sx={{
                        width: rightDrawer,
                        flexShrink: 0,
                        '& .MuiDrawer-paper': {
                            width: rightDrawer,
                            boxSizing: 'border-box',
                        },
                    }}
                    variant="persistent"
                    anchor="right"
                    open={true}
                >
                    <DrawerRight size={rightDrawer} />
                </Drawer>
            </>
        );
    }
}

export default withAuth0(withAppContext(Shell));