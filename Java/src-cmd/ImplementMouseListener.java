import java.awt.event.*;
import javax.swing.*;

public class ImplementMouseListener implements MouseListener {
	JFrame frame;

	public ImplementMouseListener() {
		frame = new JFrame();
		frame.setSize(800, 600);
		frame.setVisible(true);
		frame.addMouseListener(this);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
	public void mousePressed(MouseEvent e) {};
	public void mouseReleased(MouseEvent e) {};
	public void mouseEntered(MouseEvent e) {};
	public void mouseExited(MouseEvent e) {};
	public void mouseClicked(MouseEvent e) {
		frame.setTitle("Clicked posiont: " + e.getX() + "," + e.getY());
	}

	public static void main(String[] args) {
		new ImplementMouseListener();
	}
}