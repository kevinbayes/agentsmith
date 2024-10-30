import {
    styled, Typography,
} from "@mui/material";

export const H1Header = styled(({ ...props }) => <Typography {...props} variant={"h1"} />)(({ theme }) => ({
    '& p': {
        marginBlockStart: 0,
        marginBlockEnd: 0,
    },
    [theme.breakpoints.down('sm')]: {
        minWidth: '100%',
    }
}), { variant: "h1", });

export const H1HeaderDrawer = styled(({ ...props }) => <Typography {...props} variant={"h5"} />)(({ theme }) => ({
    '& p': {
        marginBlockStart: 0,
        marginBlockEnd: 0,
    },
    [theme.breakpoints.down('sm')]: {
        minWidth: '100%',
    }
}), { variant: "h5", });