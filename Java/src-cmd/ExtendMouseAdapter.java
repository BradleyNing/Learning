import java.awt.event.*;
import javax.swing.*;

public class ExtendMouseAdapter extends MouseAdapter {
	JFrame frame;

	public ExtendMouseAdapter() {
		frame = new JFrame();
		frame.setSize(800, 600);
		frame.setVisible(true);
		frame.addMouseListener(this);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
	public void mouseClicked(MouseEvent e) {
		frame.setTitle("Clicked posiont: " + e.getX() + "," + e.getY());
	}

	public static void main(String[] args) {
		new ExtendMouseAdapter();
	}
}